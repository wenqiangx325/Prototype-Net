import torch
import torch.nn

class Prototype(torch.nn.Module):
    """"""
    def __init__(self, prototype_num, latent_size) -> None:
        super(Prototype, self).__init__()
        self.latent_size = latent_size
        self.prototypes = torch.nn.Parameter(data=torch.random(prototype_num, latent_size), requires_grad=True)
    
    def forward(self, input):
        """"""
        x = input.view(-1, self.latent_size)
        dis_2 = self.__calc_dist_2(x, self.prototypes)
        return dis_2, self.prototypes
    
    def __calc_dist_2(self, x, y):
        """"""
        dis_2 = torch.cdist(x, y, p=2)**2
        return dis_2

