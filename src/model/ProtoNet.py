import torch
import torch.nn

from model.ConvCoder import Decoder, Encoder
from model.Prototype import Prototype

class ProtoNet(torch.nn.Module):
    """"""
    def __init__(self, prototype_number, latent_size, out_size) -> None:
        super(ProtoNet, self).__init__()
        self.encoder = Encoder(out_size=int(latent_size/4))
        self.decoder = Decoder(in_size =int(latent_size/4))
        self.prototype = Prototype(prototype_number, latent_size)
        self.fc = torch.nn.Linear(in_features=prototype_number, out_features=out_size)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, input):
        code = self.encoder(input)

        re_image = self.decoder(code)

        dis_2, prototypes = self.prototype(code)
        w = self.fc(dis_2)
        out = self.sm(w)

        return out, code, re_image, prototypes