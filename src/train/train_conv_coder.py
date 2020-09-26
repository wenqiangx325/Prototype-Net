import sys
sys.path.append("src")

import torch
import torch.optim
import torchvision.utils
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.loader import load_mnist
from model.ConvCoder import Encoder, Decoder



plt.ion()

dataroot = "data/mnist"

worker = 2

batch_size = 64

image_size = 32

num_epochs = 50

lr = 0.0002

beta1 = 0.5

ngpu = 0

dataset = load_mnist(dataroot)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

device = torch.device("cuda:0"
                      if (torch.cuda.is_available() and ngpu > 0) else "cpu")

real_batch = next(iter(dataloader))
#plt.figure(figsize=(8, 8))
#plt.axis("off")
#plt.title("Train Images")
#plt.imshow(
#    np.transpose(
#        torchvision.utils.make_grid(
#            real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),
#        (1, 2, 0)
#    )
#)
# plt.show()

encoder = Encoder().to(device=device)
if(device.type == "cuda") and (ngpu > 1):
    encoder = torch.nn.DataParallel(encoder, list(range(ngpu)))

print(encoder)

decoder = Decoder().to(device=device)
if(device.type=="cuda") and (ngpu > 1):
    decoder = torch.nn.DataParallel(decoder, list(range(ngpu)))

print(decoder)

criterion = torch.nn.MSELoss(reduction = "sum")

optimizerE = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = torch.optim.Adam(decoder.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
losses = []
iters = 0

fig = plt.figure()
loss_ax = fig.add_subplot(2,1,1)
loss_ax.set_title("loss")

real_img = fig.add_subplot(2,2,3)
real_img.set_title("real")
real_img.axis("off")

de_img = fig.add_subplot(2,2,4)
de_img.set_title("decode")
de_img.axis("off")

fig.tight_layout()

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        encoder.zero_grad()
        real_cpu = data[0].to(device)
        
        code = encoder(real_cpu)

        decoder.zero_grad()
        new_images = decoder(code)
        new_images = new_images.to(device)
        
        errN = criterion(real_cpu, new_images)
        errN.backward()
        optimizerE.step()
        optimizerD.step()

        losses.append(errN.item())

        if i % 100 == 0:
            print(losses[-1])
            loss_ax.clear()
            loss_ax.plot(losses, "r")
            real_img.imshow(real_cpu[0].detach().numpy().reshape((28,28)))
            de_img.imshow(new_images[0].detach().numpy().reshape((28,28)))
            fig.canvas.draw()
            fig.canvas.flush_events()



torch.save(encoder, "models/coder/encoder")
torch.save(decoder, "models/coder/decoder")
input()
