import torch
import matplotlib.pyplot as plt
import numpy as np
import string
import random



def get_random_string(length):
    # Random string with the combination of lower and upper case
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def plot_distribution(dataloader, what='class', title='Class Distribution of Rotated MNIST'):

    if what == 'class':
        elems = [klass for _, klass, _ in dataloader.dataset]
    elif what == 'angle':
        elems = [angle for _, _, angle in dataloader.dataset]

    unique_elems, counts = torch.unique(torch.tensor(elems), return_counts=True)

    plt.bar(unique_elems, counts, align='center', alpha=0.7)
    plt.xticks(unique_elems)
    plt.xlabel(what)
    plt.ylabel('Count')
    plt.title(title)
    plt.show()


def plot_rotated_mnist_images(dataloader, task, rotation_degree, num_images=5):
    plt.figure(figsize=(15, 3))

    b = next(iter(dataloader))[0]
    rng = np.random.default_rng()
    images = rng.choice(b, size=num_images, replace=False)

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i, 0], cmap='gray')
        plt.title(f'Angle: {rotation_degree}°')
        plt.axis('off')

    plt.show()