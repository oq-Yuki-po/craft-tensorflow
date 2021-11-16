import matplotlib.pyplot as plt
import tensorflow as tf

from src.dataset.generator import CraftDataset
from tests import OUTPUT_PATH


def test_generate_dataset():

    craft_dataset = CraftDataset()
    dataset = craft_dataset.generate()

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


def test_generate_dataset_for_icdar():
    tf.data.experimental.enable_debug_mode()
    model = tf.keras.models.load_model("results/20211109030423/saved_model")
    craft_dataset = CraftDataset(model=model)

    dataset = craft_dataset.generate(is_weak_supervised=True)

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


def test_preprocess_icdar():
    model = tf.keras.models.load_model("results/20211109030423/saved_model")
    craft_dataset = CraftDataset(model=model)
    craft_dataset.preprocess_icdar('src/dataset/icdar/train_images/img_9.jpg',
                                   'src/dataset/icdar/train_gts/gt_img_9.txt')
