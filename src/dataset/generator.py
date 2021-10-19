import pathlib

import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from imgaug.augmentables.heatmaps import HeatmapsOnImage

from src.dataset.score_genetator import generate_affinity_score, generate_region_score
from src.util import load_yaml


class CraftDataset():

    def __init__(self, is_augment: bool = True, is_semi: bool = False) -> None:
        self.is_augment = is_augment
        self.is_semi = is_semi

        cfg = load_yaml()

        self.image_height, self.image_width, _ = cfg['input_image_size']

    def generate_scores(self, image, char_bbox, text_path):
        char_bbox = char_bbox.numpy()
        text_path = text_path.numpy().decode('utf-8')
        with open(text_path, "r") as f:
            text = []
            for x in f:
                text.append(eval(x.rstrip("\n")))

        _, region = generate_region_score(image.shape, char_bbox.copy()[0])
        _, affinity = generate_affinity_score(image.shape, char_bbox.copy()[0], text)

        return region, affinity

    def preprocess_txt(self, text_path):
        text_path = text_path.numpy().decode('utf-8')
        with open(text_path, "r") as f:
            list_row = []

            for x in f:

                list_row.append(eval(x.rstrip("\n")))
        return list_row

    def preprocess_bbox(self, char_bbox_path):
        char_bbox_path = char_bbox_path.numpy().decode('utf-8')
        return np.load(char_bbox_path)

    def preprocess(self, image_path, char_box_path, text_path):
        # 画像の読み込み
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        if self.is_semi:
            pass
        else:
            # charBBoxファイルの読み込み
            char_bbox = tf.py_function(self.preprocess_bbox, [char_box_path], [tf.float32])

        scp = np.ones((self.image_height // 2, self.image_width // 2))

        scp = scp.astype(np.float32)

        region, affinity = tf.py_function(self.generate_scores, [image, char_bbox, text_path], [tf.float32, tf.float32])

        return image, region, affinity, scp

    def augment_fn(self, image, region, affinity):

        seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.3))),
            iaa.Sometimes(0.5, iaa.SomeOf(1,
                                          [iaa.HorizontalFlip(1.0),
                                           iaa.VerticalFlip(1.0)])),
            iaa.Sometimes(0.3, iaa.GaussianBlur(1.0)),
            iaa.Sometimes(0.5, iaa.Rotate(rotate=(0, 180))),
            iaa.Sometimes(0.5, iaa.AddToHue(value=(-30, 30))),
            iaa.Sometimes(0.5, iaa.AddToSaturation(value=(-30, 30))),
            iaa.Sometimes(0.5, iaa.AddToBrightness(add=(-30, 30))),
            iaa.Resize((self.image_height, self.image_width))
        ])

        region = region.astype(np.float32)
        affinity = affinity.astype(np.float32)

        region_and_affinity = np.dstack((region, affinity))

        depth_region_and_affinity = HeatmapsOnImage(
            region_and_affinity, shape=image.shape, min_value=0.0, max_value=1.0)

        aug_image, aug_heatmaps = seq(image=image, heatmaps=depth_region_and_affinity)

        aug_heatmaps = aug_heatmaps.resize((self.image_height // 2, self.image_width // 2))

        aug_heatmaps = aug_heatmaps.get_arr()

        return aug_image, aug_heatmaps

    def data_augment(self):
        def augment(image, region, affinity, scp):

            image, heatmaps = tf.numpy_function(self.augment_fn,
                                                [image, region, affinity],
                                                [tf.uint8, tf.float32])

            image = tf.dtypes.cast(image, tf.float32) / 255.0

            image.set_shape((None, None, 3))
            heatmaps.set_shape((None, None, 2))

            return image, heatmaps,scp
        return augment

    def generate(self):

        cfg = load_yaml()

        image_dir_path = f"{cfg['train_data']}/image"

        all_p_image_paths = list(pathlib.Path(image_dir_path).glob("*.jpg"))

        all_image_paths = [str(i) for i in all_p_image_paths]
        all_char_bbox_paths = [str(i).replace("image", "char_bbox").replace(".jpg", ".npy") for i in all_p_image_paths]
        all_text_paths = [str(i).replace("image", "text").replace(".jpg", ".txt") for i in all_p_image_paths]

        assert len(all_image_paths) == len(all_char_bbox_paths)
        assert len(all_image_paths) == len(all_text_paths)

        path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_char_bbox_paths, all_text_paths))

        dataset = path_ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if self.is_augment:
            dataset = dataset.map(self.data_augment(), num_parallel_calls=tf.data.AUTOTUNE)

        return dataset
