import matplotlib.pyplot as plt

from src.dataset.generator import CraftDataset
from tests.conftest import OUTPUT_PATH


def test_generate_dataset():

    craft_dataset = CraftDataset()
    dataset = craft_dataset.generate()

    plt.figure(figsize=(24, 24))
    for index, (image, heatmaps, scp) in enumerate(dataset.take(3),):
        index = index * 4
        plt.subplot(3, 4, index + 1)
        plt.imshow(image)
        plt.subplot(3, 4, index + 2)
        plt.imshow(heatmaps[:, :, 0], alpha=0.5)
        plt.subplot(3, 4, index + 3)
        plt.imshow(heatmaps[:, :, 1], alpha=0.5)
        plt.subplot(3, 4, index + 4)
        plt.imshow(scp, alpha=0.5)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f"{OUTPUT_PATH}/dataset_region_heatmap.png")
