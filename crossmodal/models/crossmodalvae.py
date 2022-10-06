from typing import Optional, Sequence
from itertools import chain
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from serotiny.models.vae.base_vae import BaseVAE
from serotiny.models.vae.priors import Prior
import torch.nn.functional as F
from omegaconf import ListConfig


from typing import Optional, Sequence
from itertools import chain
from omegaconf import DictConfig


class CrossModalVAE(BaseVAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        x_labels: list,
        latent_dim: int,
        latent_loss: dict,
        latent_loss_target: dict,
        latent_loss_weights: dict,
        latent_loss_backprop_when: dict,
        prior: dict,
        latent_loss_optimizer: dict,
        latent_loss_scheduler: dict,
        beta: float,
        optimizer: dict,
        reconstruction_loss: dict,
        lr_scheduler: dict,
        id_label: Optional[str],
        cache_outputs: Sequence = ("test",),
        **kwargs,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            x_label=x_labels[0],
            beta=beta,
            id_label=id_label,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            reconstruction_loss=reconstruction_loss,
            prior=prior,
            cache_outputs=cache_outputs,
            **kwargs,
        )
        self.x_labels = x_labels

        self.latent_loss = latent_loss
        self.latent_loss_target = latent_loss_target
        self.latent_loss_weights = latent_loss_weights
        self.latent_loss_optimizer = latent_loss_optimizer
        self.latent_loss_scheduler = latent_loss_scheduler
        self.latent_loss_backprop_when = latent_loss_backprop_when
        self.automatic_optimization = False

    def _step(self, stage, batch, batch_idx, logger):

        (
            recon_parts,
            z_parts,
            z_parts_params,
            z_composed,
            loss,
            reconstruction_loss_parts,
            kld_loss,
            kld_per_part,
        ) = self.forward(batch, decode=True, compute_loss=True)

        mu = {}
        for x_label in self.x_labels:
            mu[x_label] = z_parts_params[x_label]
            if mu[x_label].shape[1] != self.latent_dim:
                mu[x_label] = mu[x_label][:, :int(mu[x_label].shape[1]/2)]

        _loss = {}
        weighted_adv_loss = 0
        adv_loss = 0
        for part in self.latent_loss.keys():
            part_mu = mu[part]
            _loss[part] = {}

            for sub_net in self.latent_loss[part].keys():
                if isinstance(self.latent_loss_target[part][sub_net], (list, ListConfig)):
                    targets = []
                    for n in range(len(self.latent_loss_target[part][sub_net])):
                        target = torch.ones(part_mu.size()[0],1).fill_(self.latent_loss_target[part][sub_net][n])
                        targets.append(target)
                else:
                    targets = [batch[self.latent_loss_target[part][sub_net]]]

                for target in targets:
                    _loss[part][sub_net] = self.latent_loss[part][sub_net](
                        part_mu.float(), target
                    ) 
                    adv_loss += _loss[part][sub_net]
                    weighted_adv_loss += _loss[part][sub_net] * self.latent_loss_weights[part][sub_net]

        if stage != 'test':
            optimizers = self.optimizers()
            lr_schedulers = self.lr_schedulers()

            main_optim = optimizers.pop(0)
            main_lr_sched = lr_schedulers.pop(0)

            non_main_key = [i for i in self.optimizer.keys() if i != 'main']
            non_main_optims = []
            non_main_lr_scheds = []
            for i in non_main_key:
                non_main_optims.append(optimizers.pop(0))
                non_main_lr_scheds.append(lr_schedulers.pop(0))

            adversarial_flag = False
            for optim_ix, (optim, lr_sched) in enumerate(
                zip(optimizers, lr_schedulers)
            ):

                group_key = self.latent_loss_optimizer_map[optim_ix]
                mod = self.latent_loss_backprop_when.get(group_key) or 3

                if stage == 'train':
                    if batch_idx % mod == 0:
                        adversarial_flag = True
                        optim.zero_grad()
                        self.manual_backward(adv_loss)
                        optim.step()
                        # Dont use LR scheduler here, messes up the loss
                        if lr_sched is not None:
                            lr_sched.step()

            if (stage == "train") & (adversarial_flag == False):
                main_optim.zero_grad()
                for non_main_optim in non_main_optims:
                    non_main_optim.zero_grad()
                
                self.manual_backward(sum(reconstruction_loss_parts.values()) + self.beta*kld_loss + weighted_adv_loss)
                # self.manual_backward(reconstruction_loss - adv_loss)
                main_optim.step()
                for non_main_optim in non_main_optims:
                    non_main_optim.step()
                # Dont use LR scheduler here, messes up the loss
                # if main_lr_sched is not None:
                #     main_lr_sched.step()
                # for non_main_lr_sched in non_main_lr_scheds:
                #     non_main_lr_sched.step()

        loss = sum(reconstruction_loss_parts.values()) + self.beta*kld_loss + weighted_adv_loss

        results = self.make_results_dict(
            stage,
            batch,
            loss,
            reconstruction_loss_parts,
            kld_loss,
            kld_per_part,
            z_parts,
            z_parts_params,
            z_composed,
            recon_parts,
        )

        for part, value in _loss.items():
            for sub_net, sub_net_val in value.items():
                results.update(
                    {
                        f"adv_loss/{part}/{sub_net}": sub_net_val.detach().cpu(),
                    }
                )

        self.log_metrics(stage, results, logger, batch[self.x_labels[0]].shape[0])

        return results


    def configure_optimizers(self):

        get_params = lambda self: list(self.parameters())

        _parameters = get_params(self.decoder)
        for part in self.optimizer['main']['keys']:
            _parameters.extend(get_params(self.encoder[part]))

        optimizers = [self.optimizer['main']['opt'](_parameters)]

        lr_schedulers = [
            (
                self.lr_scheduler['main'](optimizer=optimizers[0])
                if self.lr_scheduler['main'] is not None
                else None
            )
        ]

        non_main_key = [i for i in self.optimizer.keys() if i != 'main']

        if len(non_main_key) > 0:
            non_main_key = non_main_key[0]
            _parameters2 = get_params(self.encoder[self.optimizer[non_main_key]['keys'][0]])
            if len(self.optimizer[non_main_key]['keys']) > 1:
                for key in self.optimizer[non_main_key]['keys'][1:]:
                    _parameters2.extend(get_params(self.encoder[self.optimizer[non_main_key]['keys'][key]]))
            optimizers.append(self.optimizer[non_main_key]['opt'](_parameters2))
            lr_schedulers.append(self.lr_scheduler[non_main_key](optimizer=optimizers[-1]))

        self.latent_loss_optimizer_map = dict()

        for optim_ix, (group_key, group) in enumerate(self.latent_loss_optimizer.items()):

            self.latent_loss_optimizer_map[optim_ix] = group_key
            _parameters3 = []
            for key in  group['keys']:
                for part in self.latent_loss[key].keys():
                    this_net = self.latent_loss[key][part]
                    this_net_params = get_params(this_net)
                    _parameters3 += this_net_params
            optimizers.append(group['opt'](_parameters3))
            lr_schedulers.append(self.latent_loss_scheduler[group_key](optimizer=optimizers[-1]))

        return optimizers, lr_schedulers