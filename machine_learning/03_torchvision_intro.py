# No entrena ninguna red neuronal, ni tampoco se crea ninguna red neuronal.
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# pip install pillow
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')


mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)


# Pick out the 46th (0-indexed) example from the training set
image, label = mnist_train[89]

arr = np.array(image)
print(f'Total de datos: {len(mnist_train)} {len(mnist_test)} Tipo de image: {type(image)} Tipo de label: {type(label)}')
print(f'Forma del array {arr.shape}')
print(f'Fila 8: {arr[0][9]}')

# Plot the image
#print(f'Default image shape: {format(image.shape)}')
image = image.reshape([28, 28])
print("Reshaped image shape: {}".format(image.shape))
plt.imshow(image, cmap="gray")

# Print the label
print(f'The label for this image: {format(label)}')

# https://stackoverflow.com/questions/42812230/why-plt-imshow-doesnt-display-the-image
plt.show()