import sys
from matplotlib.pyplot import axis
sys.path.append("src")

import torch

import numpy as np

import pylab
pylab.show()
import matplotlib.pyplot as plt

from model.ProtoNet import ProtoNet
from model.Prototype import Prototype
from model.ConvCoder import Encoder, Decoder

import utils.loader as loader

prototype_number = 15

latent_size = 40

out_size = 10

ngpu = 0

proto_net_path = "models/ProtoNet/1601186485/net.pt"

device = torch.device("cuda:0"
                      if (torch.cuda.is_available() and ngpu > 0) else "cpu")

proto_net = ProtoNet(prototype_number, latent_size, out_size)

proto_net = torch.load(proto_net_path)# proto_net.load_state_dict(torch.load(proto_net_path), device)
proto_net.to(device=device)
proto_net.eval()

# Print model's state_dict
fc_weight = None
prototypes = None
decoder_param = dict()
print("Model's state_dict:")
for param_tensor in proto_net.state_dict():
    print(param_tensor, "\t", proto_net.state_dict()[param_tensor].size())
    if "fc.weight" in param_tensor:
        fc_weight = proto_net.state_dict()["fc.weight"]
    if "prototype.prototypes" in param_tensor:
        prototypes = proto_net.state_dict()["prototype.prototypes"]

    if "decoder" in param_tensor:
        decoder_param[param_tensor.replace("decoder.", "")] = proto_net.state_dict()[param_tensor]

decoder = Decoder(int(latent_size/4))
decoder.load_state_dict(decoder_param)

prototypes_re = prototypes.view(-1,10,2,2)

images = decoder(prototypes_re)

fig = plt.figure()
for ii, img in enumerate(images):
    sub_ax = fig.add_subplot(3,5, ii+1)
    sub_ax.axis("off")
    sub_ax.imshow(img.view(28,28).detach().numpy())

fig_table = plt.figure()
for ii, img in enumerate(images):
    sub_ax = fig_table.add_subplot(11, 16, ii+2)
    sub_ax.axis("off")
    sub_ax.imshow(img.view(28,28).detach().numpy())

for ii in range(0, 10):
    sub_ax = fig_table.add_subplot(11, 16, (ii + 1)*16 + 1)
    sub_ax.axis("off")
    sub_ax.text(0.5, 0.5, ii, ha='center', va='center',fontdict={'weight':  'bold', 'size': 16})

for ii in range(0, fc_weight.shape[0]):
    for jj in range(0, fc_weight.shape[1]):
        min_j = np.argmin(fc_weight.detach().numpy()[:, jj])
        sub_ax = fig_table.add_subplot(11, 16, (ii+1)*16 + jj + 2)
        sub_ax.axis("off")

        if ii == min_j:
            sub_ax.text(0.5, 0.5, f"{fc_weight.detach().numpy()[ii][jj]:.2f}",ha='center', va='center', fontdict={'color':  'darkred', 'size':14})
        else:
            sub_ax.text(0.5, 0.5, f"{fc_weight.detach().numpy()[ii][jj]:.2f}",ha='center', va='center', fontdict={'size':14})

mnist_dataset = loader.load_mnist(train=False)

test_set = torch.tensor(mnist_dataset.data)
test_set = test_set.reshape((-1,1,28,28)).float()
test_set.to(device=device)
out, code, re_image, prototypes  = proto_net(test_set)

labels = np.argmax(out.detach().numpy(), axis=1)

labels_test = np.array(labels) == np.array(mnist_dataset.targets)

error = sum(labels_test)/len(mnist_dataset.targets)
print(f"rate: {error}")

#plt.show()

