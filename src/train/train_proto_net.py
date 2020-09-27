import sys
sys.path.append("src")

from model.ProtoNet import ProtoNet
from model.ConvCoder import Encoder, Decoder
from utils.loader import load_mnist
import numpy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.utils
import torch.nn
import torch.optim
import torch
import datetime
import pickle
import os



dataroot = "data/mnist"
modelroot = f"models/ProtoNet/{datetime.datetime.now().timestamp().__int__()}"

if not os.path.isdir(modelroot):
    os.makedirs(modelroot)

worker = 2

batch_size = 64

image_size = 32

num_epochs = 100

lr = 0.0002

beta1 = 0.5

ngpu = 0

prototype_number = 15

latent_size = 40

out_size = 10

rate1 = 0.5

rate2 = 0.5

rate3 = 0.5

dataset = load_mnist(dataroot)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

device = torch.device("cuda:0"
                      if (torch.cuda.is_available() and ngpu > 0) else "cpu")

net = ProtoNet(prototype_number, latent_size, out_size)

if(device.type == "cuda") and (ngpu > 1):
    net = torch.nn.DataParallel(net, list(range(ngpu)))

print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))


def loss_func(labels, targets, images, re_images, codes, prototypes, rate1, rate2, rate3):
    loss_cross = torch.nn.CrossEntropyLoss()
    loss_mse = torch.nn.MSELoss()
    cc = torch.stack([ torch.tensor([i]).repeat(15) for i in range(0,5)])
    a1 = torch.stack([torch.min(torch.stack([loss_mse(prototypes[ii], codes[jj]) for jj in range(codes.shape[0])]), dim=0).values for ii in range(prototypes.shape[0])])
    a2 = torch.stack([torch.min(torch.stack([loss_mse(codes[ii], prototypes[jj]) for jj in range(prototypes.shape[0])]), dim=0).values for ii in range(codes.shape[0])])


    e = torch.mean(loss_cross(labels, targets))
    # torch.mean(torch.cdist(re_images, images, p=2)**2)
    r = loss_mse(re_images, images)
    # torch.mean(torch.min(torch.cdist(prototypes, codes), dim = 0).values)
    r1 = torch.mean(torch.stack([torch.min(torch.stack([loss_mse(prototypes[ii], codes[jj]) for jj in range(codes.shape[0])]), dim=0).values for ii in range(prototypes.shape[0])]))
    # torch.mean(torch.min(torch.cdist(codes, prototypes), dim = 0).values)
    r2 = torch.mean(torch.stack([torch.min(torch.stack([loss_mse(codes[ii], prototypes[jj]) for jj in range(prototypes.shape[0])]), dim=0).values for ii in range(codes.shape[0])]))

    loss = e + rate1 * r + rate2 * r1 + rate3 * r2

    return loss


criterion = loss_func

losses = []

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        net.zero_grad()
        real_cpu = data[0].to(device)
        targets = data[1].to(device)

        out, code, re_image, prototypes = net(real_cpu)

        err_net = criterion(out, targets, real_cpu, re_image,
                            code.view(-1, latent_size), prototypes, rate1, rate2, rate3)

        err_net.backward()
        optimizer.step()

        losses.append(err_net.item())

        if i % 50 == 0 or i == len(dataloader)-1:
            print(
                f"[{epoch}/{num_epochs}] [{i}/{len(dataloader)}] loss:{err_net.item()}")
            torch.save(net, os.path.join(modelroot, "net.pt"))
            with open(os.path.join(modelroot, "losses.pk"), "wb") as f:
                pickle.dump(losses, f)

plt.plot(losses, label="loss")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend()
plt.show()
