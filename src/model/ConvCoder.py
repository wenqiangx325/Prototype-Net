import torch
import torch.nn
from torch.nn.functional import pad


class Encoder(torch.nn.Module):
    """"""

    def __init__(self, out_size=10) -> None:
        """"""
        super(Encoder, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.Sigmoid(),

            torch.nn.Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.Sigmoid(),

            torch.nn.Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.Sigmoid(),

            torch.nn.Conv2d(in_channels=32,
                            out_channels=out_size,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        """"""
        return self.main(input)


class Decoder(torch.nn.Module):
    """"""

    def __init__(self, in_size=10) -> None:
        """"""
        super(Decoder, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_size,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1),
            torch.nn.Sigmoid(),

            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0),
            torch.nn.Sigmoid(),

            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1),
            torch.nn.Sigmoid(),

            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        """"""
        return self.main(input)
