from typing import List

import numpy as np
import matplotlib.pyplot as plt


def plot_images_in_row(images: List[np.array], titles: List[str]) -> None:
    """Draws several images in horisontal row with titles

    Args:
        images (List[np.array]): list of images to plot
        titles (List[str]): list of titles for images
    """
    images_number = len(images)

    for i in range(images_number):
        plt.subplot(101 + images_number * 10 + i)
        plt.imshow(images[i], cmap = 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])


    plt.show()