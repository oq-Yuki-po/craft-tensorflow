import matplotlib.pyplot as plt

from src.dataset.generator import generate_dataset
from tests.conftest import OUTPUT_PATH


def test_generate_dataset():

    dataset = generate_dataset(is_augment=False)
    plt.figure(figsize=(24, 24))
    for index, (image, region, affinity) in enumerate(dataset.take(3),):
        index = index * 3
        plt.subplot(3, 3, index + 1)
        plt.imshow(image)
        plt.subplot(3, 3, index + 2)
        plt.imshow(image)
        plt.imshow(region, alpha=0.5)
        plt.subplot(3, 3, index + 3)
        plt.imshow(image)
        plt.imshow(affinity, alpha=0.5)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f"{OUTPUT_PATH}/dataset_region_heatmap.png")
