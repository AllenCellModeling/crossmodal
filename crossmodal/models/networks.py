import numpy as np

import torch.nn as nn

from serotiny.networks import BasicCNN

def calculate_image_size(encoder, in_channels, input_dims):

    dummy_out, intermediate_sizes = encoder.conv_forward(
        torch.zeros(1, in_channels, *input_dims), return_sizes=True
    )

    compressed_img_shape = dummy_out.shape[2:]

    intermediate_sizes = [input_dims] + intermediate_sizes[:-1]
    intermediate_sizes = intermediate_sizes[::-1]
    return intermediate_sizes, compressed_img_shape

class SymmetricImageDecoder(nn.Module):
    def __init__(
        self,
        encoder,
    ):
        super().__init__()
        
        
        intermediate_sizes, compressed_img_shape = calculate_image_size(encoder, in_channels, input_dims)
        in_channels = encoder.in_channels
        output_channels = in_channels
        input_dims = encoder.input_dims
        output_dims = tuple(input_dims)

        hidden_channels = encoder.hidden_channels
        kernel_size = encoder.kernel_size
        stride = encoder.stride
        batch_norm = encoder.batch_norm
        skip_connections = encoder.skip_connections
        non_linearity = encoder.non_linearity
        mode = encoder.mode
        max_pool_layers = encoder.max_pool_layers

        self.compressed_img_shape = compressed_img_shape
        compressed_img_size = np.prod(compressed_img_shape) * hidden_channels[0]
        orig_img_size = np.prod(output_dims)

        hidden_channels[-1] = output_channels
        self.hidden_channels = hidden_channels
        self.linear_decompress = nn.Linear(latent_dim, compressed_img_size)

        self.deconv = BasicCNN(
            hidden_channels[0],
            output_dim=orig_img_size,
            hidden_channels=hidden_channels,
            input_dims=compressed_img_shape,
            upsample_layers={
                i: tuple(size) for (i, size) in enumerate(intermediate_sizes)
            },
            up_conv=True,
            flat_output=False,
            kernel_size=kernel_size,
            mode=mode,
            non_linearity=non_linearity,
            skip_connections=skip_connections,
            batch_norm=batch_norm,
        )

    def forward(self, z):
        z = self.linear_decompress(z)
        z = z.view(
            z.shape[0],  # batch size
            self.hidden_channels[0],
            *self.compressed_img_shape
        )

        z = self.deconv(z)
        z = z.clamp(max=50)

        return z