import matplotlib.pyplot as plt
import pytest
import tensorflow as tf

from src.dataset.generator import CraftDataset
from src.model import craft
from tests import OUTPUT_PATH


@pytest.fixture()
def craft_dataset_for_synth():

    craft_dataset = CraftDataset()

    return craft_dataset


@pytest.fixture()
def craft_dataset_for_icdar(config):

    model = craft()
    latest = tf.train.latest_checkpoint(config['test_checkpoint'])
    model.load_weights(latest)

    craft_dataset = CraftDataset(model=model)

    return craft_dataset


def test_generate_dataset(craft_dataset_for_synth):

    dataset = craft_dataset_for_synth.generate()

    plt.figure(figsize=(24, 24))
    for index, (image, heatmaps) in enumerate(dataset.take(3),):
        index = index * 4
        plt.subplot(3, 4, index + 1)
        plt.imshow(image)
        plt.subplot(3, 4, index + 2)
        plt.imshow(heatmaps[:, :, 0], alpha=0.5)
        plt.subplot(3, 4, index + 3)
        plt.imshow(heatmaps[:, :, 1], alpha=0.5)
        plt.subplot(3, 4, index + 4)
        plt.imshow(heatmaps[:, :, 2], alpha=0.5)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f"{OUTPUT_PATH}/dataset_region_heatmap.png")


def test_generate_dataset_for_icdar(craft_dataset_for_icdar):

    dataset = craft_dataset_for_icdar.generate(is_weak_supervised=True)

    plt.figure(figsize=(24, 24))
    for index, (image, heatmaps) in enumerate(dataset.take(3),):
        index = index * 4
        plt.subplot(3, 4, index + 1)
        plt.imshow(image)
        plt.subplot(3, 4, index + 2)
        plt.imshow(heatmaps[:, :, 0], alpha=0.5)
        plt.subplot(3, 4, index + 3)
        plt.imshow(heatmaps[:, :, 1], alpha=0.5)
        plt.subplot(3, 4, index + 4)
        plt.imshow(heatmaps[:, :, 2], alpha=0.5)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f"{OUTPUT_PATH}/dataset_region_heatmap.png")


def test_preprocess_icdar(craft_dataset_for_icdar):

    craft_dataset_for_icdar.preprocess_icdar('src/dataset/icdar/train_images/img_9.jpg',
                                             'src/dataset/icdar/train_gts/gt_img_9.txt')
