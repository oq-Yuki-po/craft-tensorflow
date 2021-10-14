import pathlib

import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from imgaug.augmentables.heatmaps import HeatmapsOnImage

from src.dataset.score_genetator import generate_affinity_score, generate_region_score
from src.util import load_yaml


def generate_scores(image, char_bbox, text_path):
    char_bbox = char_bbox.numpy()
    text_path = text_path.numpy().decode('utf-8')
    with open(text_path, "r") as f:
        text = []
        for x in f:
            text.append(eval(x.rstrip("\n")))

    _, region = generate_region_score(image.shape, char_bbox.copy()[0])
    _, affinity = generate_affinity_score(image.shape, char_bbox.copy()[0], text)

    return region, affinity


def preprocess_txt(text_path):
    text_path = text_path.numpy().decode('utf-8')
    with open(text_path, "r") as f:
        list_row = []

        for x in f:

            list_row.append(eval(x.rstrip("\n")))
    return list_row


def preprocess_bbox(char_bbox_path):
    char_bbox_path = char_bbox_path.numpy().decode('utf-8')
    return np.load(char_bbox_path)


def preprocess(image_path, char_box_path, text_path):
    # 画像の読み込み
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # charBBoxファイルの読み込み
    char_bbox = tf.py_function(preprocess_bbox, [char_box_path], [tf.float32])

    region, affinity = tf.py_function(generate_scores, [image, char_bbox, text_path], [tf.float32, tf.float32])

    return image, region, affinity


def augment_fn(image, region, affinity):

    seq = iaa.Sequential([
        iaa.Sharpen((0.0, 1.0)),       # sharpen the image
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects heatmaps)
    ], random_order=True)

    region = region.astype(np.float32)
    affinity = affinity.astype(np.float32)

    region_and_affinity = np.dstack((region, affinity))

    depth_region_and_affinity = HeatmapsOnImage(region_and_affinity, shape=image.shape, min_value=0.0, max_value=1.0)

    aug_image, aug_heatmaps = seq(image=image, heatmaps=depth_region_and_affinity)


    aug_heatmaps = aug_heatmaps.resize((384, 384))

    aug_heatmaps = aug_heatmaps.get_arr()

    aug_region, aug_affinity = np.dsplit(aug_heatmaps, 2)

    return aug_image, aug_region[:, :, 0], aug_affinity[:, :, 0]


def data_augment():
    def augment(image, region, affinity):

        image, region, affinity = tf.numpy_function(augment_fn,
                                                    [image, region, affinity],
                                                    [tf.uint8, tf.float32, tf.float32])

        image = tf.dtypes.cast(image, tf.float32) / 255.0

        region.set_shape((None, None))
        affinity.set_shape((None, None))
        image.set_shape((None, None, 3))

        return image, region, affinity
    return augment


def generate_dataset(is_augment: bool = True):

    cfg = load_yaml()

    image_dir_path = f"{cfg['train_data']}/image"

    all_p_image_paths = list(pathlib.Path(image_dir_path).glob("*.jpg"))

    all_image_paths = [str(i) for i in all_p_image_paths]
    all_char_bbox_paths = [str(i).replace("image", "char_bbox").replace(".jpg", ".npy") for i in all_p_image_paths]
    all_text_paths = [str(i).replace("image", "text").replace(".jpg", ".txt") for i in all_p_image_paths]

    assert len(all_image_paths) == len(all_char_bbox_paths)
    assert len(all_image_paths) == len(all_text_paths)

    path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_char_bbox_paths, all_text_paths))

    dataset = path_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if is_augment:
        dataset = dataset.map(data_augment(), num_parallel_calls=tf.data.AUTOTUNE)

    return dataset
