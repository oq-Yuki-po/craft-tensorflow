import pathlib
import random

import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from imgaug.augmentables.heatmaps import HeatmapsOnImage

from src.dataset.score_genetator import generate_affinity_score, generate_region_score
from src.util import load_yaml

random.seed(66)


class CraftDataset():

    def __init__(self, is_augment: bool = True, is_semi: bool = False) -> None:
        """初期化

        Args:
            is_augment (bool, optional): データ拡張を行うか判定. Defaults to True.
            is_semi (bool, optional): 弱教師あり学習用にデータを生成するか判定. Defaults to False.
        """
        self.is_augment = is_augment
        self.is_semi = is_semi

        self.cfg = load_yaml()

        self.input_image_height, self.input_image_width, _ = self.cfg['input_image_size']

        # データ拡張の定義
        self.seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.3))),
            iaa.Sometimes(0.5, iaa.SomeOf(1,
                                          [iaa.HorizontalFlip(1.0),
                                           iaa.VerticalFlip(1.0)])),
            iaa.Sometimes(0.3, iaa.GaussianBlur(1.0)),
            iaa.Sometimes(0.5, iaa.Rotate(rotate=(0, 180))),
            iaa.Sometimes(0.5, iaa.AddToHue(value=(-30, 30))),
            iaa.Sometimes(0.5, iaa.AddToSaturation(value=(-30, 30))),
            iaa.Sometimes(0.5, iaa.AddToBrightness(add=(-30, 30))),
            iaa.Resize((self.input_image_height, self.input_image_width))
        ])

    def generate_scores(self, image: tf.Tensor, char_bbox: tf.Tensor, text_path: tf.Tensor):
        """[summary]

        Args:
            image (tf.Tensor): 画像
            char_bbox (tf.Tensor): 文字単位のBBox
            text_path (tf.Tensor): 文字のファイルパス

        Returns:
            Tuple[np.ndarray, np.ndarray]: Region Score, Affinity Score
        """

        # numpyに変換
        char_bbox = char_bbox.numpy()
        text_path = text_path.numpy().decode('utf-8')

        # テキストから文字単位に分割
        with open(text_path, "r") as f:
            text = []
            for x in f:
                text.append(eval(x.rstrip("\n")))

        _, region = generate_region_score(image.shape, char_bbox.copy()[0])
        _, affinity = generate_affinity_score(image.shape, char_bbox.copy()[0], text)

        return region, affinity

    def generate_scp(self, image: tf.Tensor):
        """信頼度マップの生成

        Args:
            image (tf.Tensor): 画像

        Returns:
            [np.ndarray]: 信頼度マップ
        """

        scp = np.ones((image.shape[0], image.shape[1]))

        return scp

    def preprocess_bbox(self, char_bbox_path: tf.Tensor):
        """文字単位のBBoxのファイル読み込み

        Args:
            char_bbox_path (tf.Tensor): 文字単位のBBoxのファイルパス

        Returns:
            [np.ndarray]: 文字単位のBBox
        """
        char_bbox_path = char_bbox_path.numpy().decode('utf-8')
        return np.load(char_bbox_path)

    def preprocess(self, image_path: tf.Tensor, char_box_path: tf.Tensor, text_path: tf.Tensor):
        """Dataset作成の前処理

        Args:
            image_path (tf.Tensor): 画像パス
            char_box_path (tf.Tensor): 文字単位のBBoxのファイルパス
            text_path (tf.Tensor): テキストファイルのファイルパス

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 画像, Region Score, Affinity Score, 信頼度マップ
        """

        # 画像の読み込み
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=3)

        scp = tf.py_function(self.generate_scp, [image], [tf.float32])

        if self.is_semi:
            pass
        else:
            # charBBoxファイルの読み込み
            char_bbox = tf.py_function(self.preprocess_bbox, [char_box_path], [tf.float32])

        region, affinity = tf.py_function(self.generate_scores, [image, char_bbox, text_path], [tf.float32, tf.float32])

        return image, region, affinity, scp[0]

    def augment_fn(self, image: np.ndarray, region: np.ndarray, affinity: np.ndarray, scp: np.ndarray):
        """データ拡張処理

        Args:
            image (np.ndarray): 画像
            region (np.ndarray): Region Score
            affinity (np.ndarray): Affinity Score
            scp (np.ndarray): 信頼度マップ

        Returns:
            Tuple[np.ndarray, np.ndarray]: 拡張された画像, (拡張されたRegion Score, Affinity Score, 信頼度マップ)
        """

        region = region.astype(np.float32)
        affinity = affinity.astype(np.float32)

        heatmaps = np.dstack((region, affinity, scp))

        depth_heatmaps = HeatmapsOnImage(heatmaps, shape=image.shape, min_value=0.0, max_value=1.0)

        aug_image, aug_heatmaps = self.seq(image=image, heatmaps=depth_heatmaps)

        aug_heatmaps = aug_heatmaps.resize((self.input_image_height // 2, self.input_image_width // 2))

        aug_heatmaps = aug_heatmaps.get_arr()

        return aug_image, aug_heatmaps

    def data_augment(self):
        def augment(image, region, affinity, scp):

            image, heatmaps = tf.numpy_function(self.augment_fn,
                                                [image, region, affinity, scp],
                                                [tf.uint8, tf.float32])

            image = tf.dtypes.cast(image, tf.float32) / 255.0

            image.set_shape((None, None, 3))
            heatmaps.set_shape((None, None, 3))

            return image, heatmaps
        return augment

    def generate(self):

        image_dir_path = f"{self.cfg['train_data']}/image"

        all_p_image_paths = list(pathlib.Path(image_dir_path).glob("*.jpg"))

        random.shuffle(all_p_image_paths)

        all_image_paths = [str(i) for i in all_p_image_paths]
        all_char_bbox_paths = [str(i).replace("image", "char_bbox").replace(".jpg", ".npy") for i in all_p_image_paths]
        all_text_paths = [str(i).replace("image", "text").replace(".jpg", ".txt") for i in all_p_image_paths]

        assert len(all_image_paths) == len(all_char_bbox_paths)
        assert len(all_image_paths) == len(all_text_paths)

        path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_char_bbox_paths, all_text_paths)).\
            shuffle(buffer_size=self.cfg['train_shuffle_buffer_size'], seed=self.cfg['train_shuffle_seed'])

        dataset = path_ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if self.is_augment:
            dataset = dataset.map(self.data_augment(), num_parallel_calls=tf.data.AUTOTUNE)

        return dataset
