# TODO: cnn for processing image
import torch
import torch.nn as nn


class ConvolutionalNet(nn.Module):
    def __init__(self, image_width: int, num_channels: int, num_conv_channels: int, kernel_size: int,
                 dropout_probability: float, output_dimension: int, max_pool_kernel_size: int, max_pool_stride: int,
                 intermediate_hidden_size: int,
                 num_padding=0, stride=1):
        super(ConvolutionalNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout2d(dropout_probability)
        self.relu = nn.ReLU()
        self.max_pool_2d = nn.MaxPool2d(kernel_size=max_pool_kernel_size, stride=max_pool_stride)
        output_dim = (image_width - kernel_size + 2 * num_padding) / stride + 1
        dilation = 1
        second_output_dim = int(
            (output_dim + 2 * num_padding - dilation * (max_pool_stride - 1) - 1) / max_pool_stride + 1)
        self.fully_connected_1 = nn.Linear(second_output_dim * second_output_dim * num_conv_channels,
                                           intermediate_hidden_size)
        self.fully_connected_2 = nn.Linear(intermediate_hidden_size, output_dimension)
        layers = [self.conv, self.relu, self.max_pool_2d]
        self.layers = nn.Sequential(*layers)

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        batch_size = input_images.size(0)
        input_images = input_images.transpose(1, 3)
        images_features = self.layers(input_images)
        images_features = self.dropout(images_features)
        images_features = self.fully_connected_1(images_features.view(batch_size, -1))
        images_features = self.dropout(images_features)
        output_features = self.fully_connected_2(images_features)
        return output_features
