from typing import Optional
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from skimage import measure as skmeasure
from skimage import morphology as skmorpho
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk
# from pyefd import elliptic_fourier_descriptors,reconstruct_contour
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import tempfile
import os

from aicscytoparam import cytoparam
from aicsshparam import shtools

def filter_columns(
    cols_to_filter,
    regex=None,
    startswith=None,
    endswith=None,
    contains=None,
    excludes=None,
):
    if regex is not None:
        return [col for col in cols_to_filter if re.match(regex, col)]

    keep = [True] * len(cols_to_filter)
    for i in range(len(cols_to_filter)):
        if startswith is not None:
            keep[i] &= str(cols_to_filter[i]).startswith(startswith)
        if endswith is not None:
            keep[i] &= str(cols_to_filter[i]).endswith(endswith)
        if contains is not None:
            keep[i] &= contains in str(cols_to_filter[i])
        if excludes is not None:
            keep[i] &= excludes not in str(cols_to_filter[i])

    return [col for col, keep_col in zip(cols_to_filter, keep) if keep_col]

def get_ranked_dims(
    stats,
    cutoff_kld_per_dim,
    max_num_shapemodes,
):

    stats = (
        stats.loc[stats["test_kld_per_latent_dim"] > cutoff_kld_per_dim]
        .sort_values(by=["test_kld_per_latent_dim"])
        .reset_index(drop=True)
    )

    ranked_z_dim_list = stats["dimension"][::-1].tolist()
    mu_std_list = stats["mu_std_per_latent_dim"][::-1].tolist()
    mu_mean_list = stats["mu_mean_per_latent_dim"][::-1].tolist()

    if len(ranked_z_dim_list) > max_num_shapemodes:
        ranked_z_dim_list = ranked_z_dim_list[:max_num_shapemodes]
        mu_std_list = mu_std_list[:max_num_shapemodes]

    return ranked_z_dim_list, mu_std_list, mu_mean_list

class LatentWalk(Callback):
    """"""

    def __init__(
        self,
        embedding_dim: int,
        spharm_cols_filter: list,
        x_label: str,
        latent_walk_range: Optional[list] = None,
        cutoff_kld_per_dim: Optional[float] = None,
        plot_limits: Optional[list] = None,
        input_mode: Optional[str] = 'spharm',
        compute_features: Optional[bool] = True,
        max_num_shapemodes: Optional[int] = 20,
    ):
        """
        Args:
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        self.cutoff_kld_per_dim = cutoff_kld_per_dim
        self.latent_walk_range = latent_walk_range
        self.plot_limits = plot_limits
        self.spharm_cols_filter = spharm_cols_filter
        self.input_mode = input_mode
        self.compute_features = compute_features
        self.max_num_shapemodes = max_num_shapemodes
        self.x_label = x_label

        if self.latent_walk_range is None:
            self.latent_walk_range = [
                -2,
                -1,
                -0.5,
                -0.25,
                -0.1,
                0,
                0.1,
                0.25,
                0.5,
                1,
                2,
            ]

        if self.cutoff_kld_per_dim is None:
            self.cutoff_kld_per_dim = 0

        if self.plot_limits is None:
            # self.plot_limits = [-150, 150, -80, 80]
            self.plot_limits = [-120, 120, -140, 140]

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        with torch.no_grad():

            with tempfile.TemporaryDirectory() as tmp_dir:
                client = mlflow.tracking.MlflowClient(mlflow.get_tracking_uri())
                stats = pd.read_csv(
                    client.download_artifacts(
                        run_id=mlflow.active_run().info.run_id,
                        path="dataframes/stats_per_dim_test.csv",
                        dst_path=tmp_dir
                    )
                )
            ranked_z_dim_list, mu_std_list, mu_mean_list = get_ranked_dims(
                stats, self.cutoff_kld_per_dim,
                max_num_shapemodes=pl_module.latent_dim
            )

            batch_size = trainer.test_dataloaders[0].batch_size
            test_dataloader = trainer.test_dataloaders[0]
            test_iter = next(iter(test_dataloader))

            if self.input_mode == 'spharm':
                dna_spharm_cols = filter_columns(
                    trainer.datamodule.datasets["train"].columns,
                    **self.spharm_cols_filter
                )
            else:
                dna_spharm_cols = None

            compute_projections(
                pl_module,
                ranked_z_dim_list,
                mu_std_list,
                mu_mean_list,
                batch_size,
                self.embedding_dim,
                self.latent_walk_range,
                dna_spharm_cols,
                self.spharm_cols_filter,
                self.input_mode,
                self.max_num_shapemodes,
                self.plot_limits,
                self.compute_features,
                self.x_label,
            )

def compute_projections(
    pl_module: LightningModule,
    ranked_z_dim_list: list,
    mu_std_list: list,
    mu_mean_list: list,
    batch_size: int,
    latent_dims: int,
    latent_walk_range: list,
    dna_spharm_cols: list,
    spharm_cols_filter: dict,
    input_mode: str,
    max_num_shapemodes: int,
    plot_limits: list,
    compute_features: bool,
    x_label: str,
):

    matplotlib.rc("xtick", labelsize=3)
    matplotlib.rc("ytick", labelsize=3)
    matplotlib.rcParams["xtick.major.size"] = 0.1
    matplotlib.rcParams["xtick.major.width"] = 0.1
    matplotlib.rcParams["xtick.minor.size"] = 0.1
    matplotlib.rcParams["xtick.minor.width"] = 0.1

    matplotlib.rcParams["ytick.major.size"] = 0.1
    matplotlib.rcParams["ytick.major.width"] = 0.1
    matplotlib.rcParams["ytick.minor.size"] = 0.1
    matplotlib.rcParams["ytick.minor.width"] = 0.1

    # get mu means NOT ORDERED by rank
    mu_means = torch.tensor(
        [mu_mean_list[ranked_z_dim_list.index(_)]
        for _ in range(latent_dims)]
    )

    all_features_df = []
    for rank, z_dim in enumerate(ranked_z_dim_list):
        if rank == max_num_shapemodes:
            break
        # Set subplots
        fig, ax_array = plt.subplots(
            3,
            len(latent_walk_range),
            squeeze=False,
            figsize=(15, 5),
        )

        for value_index, value in enumerate(latent_walk_range):
            z_inf = torch.zeros(batch_size, latent_dims) + mu_means

            z_inf[:, z_dim] += value * mu_std_list[rank]
            z_inf = z_inf.to(pl_module.device)
            z_inf = z_inf.float()
            decoder = pl_module.decoder[x_label]

            proj_list = [0, 1, 2]
            if input_mode == 'spharm':
                x_hat = decoder(z_inf).cpu()
                img = get_image_from_shcoeffs(x_hat[0, :], spharm_cols_filter, dna_spharm_cols)
            # elif input_mode == 'fourier':
            #     proj_list = [0]
            #     x_hat = decoder(z_inf).cpu()
            #     x_hat = x_hat[0,:].cpu().numpy().reshape(4,-1)
            #     img = get_image_from_fourier(x_hat, 200, 100)
            #     plot_limits = [-S, S, -S, S]
            else:
                img = decoder(z_inf).cpu()
                img = np.array(img[0,0,:,:,:])
                plot_limits = [0, img.shape[1], 0, img.shape[2]]

            if compute_features:
                features = get_basic_features(img)
                # features['shapemode'] = rank
                features["sigma"] = value
                features[f"sigma_{rank+1}"] = value
                features[f"dim rank"] = rank+1
                this_feature_df = pd.DataFrame.from_dict(
                    features, orient="index", columns=["value"]
                )
                this_feature_df = pd.DataFrame(features, index=[0])
                all_features_df.append(this_feature_df)

            for proj in proj_list:
                plt.style.use("dark_background")
                ax_array[proj, value_index].imshow(img.max(proj), cmap="gray")
                ax_array[proj, value_index].set_title(
                    f"{value}" r"$\sigma$", fontsize=14
                )
                ax_array[proj, value_index].set_xlim([0, plot_limits[1]])
                ax_array[proj, value_index].set_ylim([0, plot_limits[3]])
                for tick in ax_array[proj, value_index].xaxis.get_major_ticks():
                    tick.label.set_fontsize(5)
                for tick in ax_array[proj, value_index].yaxis.get_major_ticks():
                    tick.label.set_fontsize(5)
                plt.style.use("default")
        print(f"Finished latent walk for dim {z_dim}, rank {rank}")
        # [ax.axis("off") for ax in ax_array.flatten()]
        # Save figure
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = os.path.join(tmp_dir, f"dim_{z_dim}_rank_{rank + 1}.png")
            ax_array.flatten()[0].get_figure().savefig(dest_path, dpi=300, bbox_inches="tight")

            mlflow.log_artifact(
                local_path=dest_path,
                artifact_path="images"
            )

    if compute_features:
        all_features = pd.concat(all_features_df, axis=0).reset_index(drop=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = Path(tmp_dir) / "all_features.csv"
            all_features.to_csv(dest_path, index=False)
            mlflow.log_artifact(
                local_path=dest_path,
                artifact_path="dataframes"
            )

            g = sns.pairplot(
                 all_features,
                 x_vars=[f"sigma_{i+1}" for i in range(5)],
                 y_vars=["shape_volume_lcc", "position_depth_lcc",
                         "roundness_surface_area_lcc",
                         "position_x_centroid_lcc", "position_y_centroid_lcc",
                         "position_z_centroid_lcc",]
            )

            dest_path = os.path.join(tmp_dir, "scatterplot_all_features.png")
            g.figure.savefig(dest_path, dpi=300, bbox_inches="tight")

            mlflow.log_artifact(
                local_path=dest_path,
                artifact_path="images"
            )

            plt.close(g.figure)


# def get_image_from_fourier(x, S, num_points):
#     contour_r = reconstruct_contour(x, locus=(0, 0), num_points=num_points)
#     contour_r = S/4 * contour_r + S/2
#     img = np.zeros((S, S),dtype='uint8')
#     img = cv2.fillPoly(img, pts=[contour_r.astype('int64')], color=(255, 255, 255))
#     img = img.reshape(-1, img.shape[0], img.shape[1])
#     return img


def get_image_from_shcoeffs(x, spharm_cols_filter, spharm_cols):
    try:
        x = x.detach().cpu().numpy()
    except:
        pass
    x = pd.DataFrame(x).T
    x.columns = spharm_cols
    x = x.iloc[0]

    mesh = get_mesh_from_series(
        x,
        spharm_cols_filter['startswith'][:-1], 
        32
    )
    # Find mesh coordinates
    coords = vtk_to_numpy(mesh.GetPoints().GetData())

    # Find bounds of the mesh
    rmin = (coords.min(axis=0) - 0.5).astype(np.int)
    rmax = (coords.max(axis=0) + 0.5).astype(np.int)
    # Create image data
    imagedata = vtk.vtkImageData()
    w, h, d = 150, 150, 150
    imagedata.SetDimensions([w, h, d])
    imagedata.SetExtent(0, w - 1, 0, h - 1, 0, d - 1)
    imagedata.SetOrigin(rmin)
    imagedata.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    # Set all values to 1
    imagedata.GetPointData().GetScalars().FillComponent(0, 1)

    # Create an empty 3D numpy array to sum up
    # voxelization of all meshes
    img = np.zeros((d, h, w), dtype=np.uint8)

    # Voxelize mesh
    seg = cytoparam.voxelize_mesh(
        imagedata=imagedata, shape=(d, h, w), mesh=mesh, origin=rmin
    )
    img[seg > 0] = 1
    return img

def get_basic_features(img):
    features = {}
    input_image = img.copy()
    input_image = (input_image > 0.5).astype(np.uint8)
    input_image_lcc = skmeasure.label(input_image)
    features[f"connectivity_cc"] = input_image_lcc.max()
    if features[f"connectivity_cc"] > 0:
        counts = np.bincount(input_image_lcc.reshape(-1))
        lcc = 1 + np.argmax(counts[1:])
        input_image_lcc[input_image_lcc != lcc] = 0
        input_image_lcc[input_image_lcc == lcc] = 1
        input_image_lcc = input_image_lcc.astype(np.uint8)
        for img, suffix in zip([input_image, input_image_lcc], ["", "_lcc"]):
            z, y, x = np.where(img)
            features[f"shape_volume{suffix}"] = img.sum()
            features[f"position_depth{suffix}"] = 1 + np.ptp(z)
            for uname, u in zip(["x", "y", "z"], [x, y, z]):
                features[f"position_{uname}_centroid{suffix}"] = u.mean()
            features[f"roundness_surface_area{suffix}"] = get_surface_area(img)
    else:
        for img, suffix in zip([input_image, input_image_lcc], ["", "_lcc"]):
            features[f"shape_volume{suffix}"] = np.nan
            features[f"position_depth{suffix}"] = np.nan
            for uname in ["x", "y", "z"]:
                features[f"position_{uname}_centroid{suffix}"] = np.nan
            features[f"roundness_surface_area{suffix}"] = np.nan
    return features


def get_surface_area(input_img):
    # Forces a 1 pixel-wide offset to avoid problems with binary
    # erosion algorithm
    input_img[:, :, [0, -1]] = 0
    input_img[:, [0, -1], :] = 0
    input_img[[0, -1], :, :] = 0
    input_img_surface = np.logical_xor(
        input_img, skmorpho.binary_erosion(input_img)
    ).astype(np.uint8)
    # Loop through the boundary voxels to calculate the number of
    # boundary faces. Using 6-neighborhod.
    pxl_z, pxl_y, pxl_x = np.nonzero(input_img_surface)
    dx = np.array([0, -1, 0, 1, 0, 0])
    dy = np.array([0, 0, 1, 0, -1, 0])
    dz = np.array([-1, 0, 0, 0, 0, 1])
    surface_area = 0
    for (k, j, i) in zip(pxl_z, pxl_y, pxl_x):
        surface_area += 6 - input_img_surface[k + dz, j + dy, i + dx].sum()
    return int(surface_area)

def get_mesh_from_series(row, alias, lmax):
    coeffs = np.zeros((2, lmax, lmax), dtype=np.float32)
    for l in range(lmax):
        for m in range(l + 1):
            try:
                # Cosine SHE coefficients
                coeffs[0, l, m] = row[
                    [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}C" in f]
                ]
                # Sine SHE coefficients
                coeffs[1, l, m] = row[
                    [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}S" in f]
                ]
            # If a given (l,m) pair is not found, it is assumed to be zero
            except:
                pass
    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs)
    return mesh