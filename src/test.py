import random
import matplotlib.pyplot as plt

import utils.loader as loader

# test load MNIST dataset by utils.loader.load_mnist
mnist_dataset = loader.load_mnist()

# print mnist_data information
print(mnist_dataset)

# show 9 random images and label from mnist_data
fig, axs = plt.subplots(3,3)
for ii, item in enumerate(random.sample(range(0, len(mnist_dataset.data)), 9)):
    axs[int(ii%3), int(ii/3)].imshow(mnist_dataset.data[item])
    axs[int(ii%3), int(ii/3)].set_title(f"{mnist_dataset.targets[item]}")
plt.show()
