import pathlib
import random

import cv2
import imgaug.augmenters as iaa
import numpy as np
import Polygon as plg
import tensorflow as tf
from imgaug.augmentables.heatmaps import HeatmapsOnImage

from src.dataset.score_genetator import generate_affinity_score, generate_region_score
from src.dataset.watershed import watershed
from src.util import load_yaml, normalizeMeanVariance, resize_aspect_ratio

random.seed(66)


class CraftDataset():

    def __init__(self, is_augment: bool = True, model: tf.keras.Model = None) -> None:
        """初期化

        Args:
            is_augment (bool, optional): データ拡張を行うか判定. Defaults to True.
            model (tf.keras.Model, optional): 弱教師あり学習に使うモデル. Defaults to None.
        """
        self.is_augment = is_augment
        self.model = model

        self.cfg = load_yaml()

        self.input_image_height, self.input_image_width, _ = self.cfg['input_image_size']

        # データ拡張の定義
        self.seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.3))),
            iaa.Sometimes(0.5, iaa.SomeOf(1,
                                          [iaa.HorizontalFlip(1.0),
                                           iaa.VerticalFlip(1.0)])),
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 3.0))),
            iaa.Sometimes(0.5, iaa.Rotate(rotate=(-30, 30))),
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
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        scp = tf.ones([image_height, image_width])

        # charBBoxファイルの読み込み
        char_bbox = tf.py_function(self.preprocess_bbox, [char_box_path], [tf.float32])

        region, affinity = tf.py_function(self.generate_scores, [image, char_bbox, text_path], [tf.float32, tf.float32])

        return image, region, affinity, scp

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

    def preprocess_icdar(self, image_path: tf.Tensor, gt_path: tf.Tensor):

        # 画像の読み込み
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=3)
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        scp = tf.ones([image_height, image_width])

        # 単語単位のbboxとテキストの取得
        word_bboxes, texts = tf.py_function(self.preprocess_word_bbox_text, [gt_path], [tf.float32, tf.string])

        region, affinity, scp = tf.py_function(self.generate_socore_map,
                                               [image, word_bboxes, texts, scp],
                                               [tf.float32, tf.float32, tf.float32])

        return image, region, affinity, scp

    def preprocess_word_bbox_text(self, gt_path):
        bboxes = []
        texts = []
        with open(gt_path.numpy().decode('utf-8'), "r") as f:
            for line in f:
                bbox = [int(i) for i in line.split(",")[:8]]
                text = line.split(",")[-1]
                texts.append(text.replace("\n", ""))
                for i in bbox:
                    assert type(i) == int
                bbox = np.array(bbox)
                bboxes.append(bbox.reshape(4, 2))
        return np.stack(bboxes, 0), texts

    def generate_socore_map(self, image, word_bboxes, texts, scp):
        pseudo_char_boxes = []
        scp = scp.numpy()
        texts = [i.numpy().decode('utf-8') for i in texts]
        new_texts = []
        for word_bbox, text in zip(word_bboxes, texts):

            # text = text.numpy().decode('utf-8')

            x_max, x_min = int(np.max(word_bbox[:, 0])), int(np.min(word_bbox[:, 0]))
            y_max, y_min = int(np.max(word_bbox[:, 1])), int(np.min(word_bbox[:, 1]))

            if text == "###":
                scp[y_min:y_max, x_min:x_max] = 0

            cropped_image = image[y_min:y_max, x_min:x_max]
            if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                continue

            # 前処理
            image_resized, _, _ = resize_aspect_ratio(cropped_image.numpy(), 640, cv2.INTER_LINEAR)
            image_resized = normalizeMeanVariance(image_resized)
            inputs = np.expand_dims(image_resized, 0)
            # 推論
            heatmaps = self.model.predict_on_batch(inputs)
            # region scoreのみ取得
            region_score = heatmaps[0, :, :, 0]
            region_score = (region_score * 255).astype(np.uint8)
            # 元の画像のサイズにリサイズ
            region_score = cv2.resize(region_score, (cropped_image.shape[1], cropped_image.shape[0]))
            region_score_image = cv2.applyColorMap(region_score, cv2.COLORMAP_JET)
            pseudo_char_boxes_per_word = watershed(cropped_image.numpy(), region_score_image)

            confidence = self.get_confidence(len(text), len(pseudo_char_boxes_per_word))

            if confidence < 0.5:
                char_bboxes = []
                # 0.5以下なら真の文字数でbboxを均等に分割する
                width = cropped_image.shape[1]
                height = cropped_image.shape[0]

                width_per_char = width / len(text)
                for i, char in enumerate(text):
                    if char == ' ':
                        continue
                    left = i * width_per_char
                    right = (i + 1) * width_per_char
                    bbox = np.array([[left, 0],
                                     [right, 0],
                                     [right, height],
                                     [left, height]])
                    char_bboxes.append(bbox)

                pseudo_char_boxes_per_word = np.array(char_bboxes, np.float32)
                confidence = 0.5
            scp[y_min:y_max, x_min:x_max] = confidence
            pseudo_char_boxes_per_word[:, :, 0] = pseudo_char_boxes_per_word[:, :, 0] + x_min
            pseudo_char_boxes_per_word[:, :, 1] = pseudo_char_boxes_per_word[:, :, 1] + y_min
            pseudo_char_boxes.append(pseudo_char_boxes_per_word)
            new_texts.append("a" * len(pseudo_char_boxes_per_word))

        pseudo_char_boxes = np.concatenate(pseudo_char_boxes)
        _, region = generate_region_score(image.shape, pseudo_char_boxes.copy())
        _, affinity = generate_affinity_score(image.shape, pseudo_char_boxes.copy(), new_texts)

        return region, affinity, scp

    def get_confidence(self, real_len, pseudo_len):
        """真の文字数と推論した文字数を比較して信頼度を算出
        """
        if pseudo_len == 0:
            return 0.
        return (real_len - min(real_len, abs(real_len - pseudo_len))) / real_len

    def generate(self, is_weak_supervised=False):
        """

        Args:
            is_weak_supervised (bool): 弱教師あり学習Dataset作成か判定. Defaults to True.

        Returns:
            tf.data.Dataset
        """
        if is_weak_supervised:
            if self.model is None:
                raise ValueError("error!")

            all_image_paths = []
            all_gt_paths = []

            icdar_dir_path = f"{self.cfg['train_data_icdar']}"
            with open(f"{icdar_dir_path}/train_list.txt", "r") as f:
                image_names = f.read().split("\n")
            for image_name in image_names:
                image_path = f"{icdar_dir_path}/train_images/{image_name}"
                gt_path = f"{icdar_dir_path}/train_gts/gt_{image_name.replace('.jpg', '.txt')}"

                p_image_path = pathlib.Path(image_path)
                p_gt_path = pathlib.Path(gt_path)

                assert p_image_path.exists() == True
                assert p_gt_path.exists() == True

                all_image_paths.append(str(p_image_path))
                all_gt_paths.append(str(p_gt_path))

            assert len(all_image_paths) == len(all_gt_paths)

            c = list(zip(all_image_paths, all_gt_paths))
            random.shuffle(c)
            all_image_paths, all_gt_paths = zip(*c)

            path_ds = tf.data.Dataset.from_tensor_slices((list(all_image_paths), list(all_gt_paths)))

            dataset = path_ds.map(self.preprocess_icdar, num_parallel_calls=tf.data.AUTOTUNE)

            if self.is_augment:
                dataset = dataset.map(self.data_augment(), num_parallel_calls=tf.data.AUTOTUNE)

        else:
            # 画像パスの読み込み
            image_dir_path = f"{self.cfg['train_data']}/image"

            all_p_image_paths = list(pathlib.Path(image_dir_path).glob("*.jpg"))

            all_p_image_paths = all_p_image_paths[:int(
                len(all_p_image_paths) * self.cfg['train_synth_data_percentage_to_use'])]

            random.shuffle(all_p_image_paths)

            all_image_paths = [str(i) for i in all_p_image_paths]
            all_char_bbox_paths = [str(i).replace("image", "char_bbox").replace(".jpg", ".npy")
                                   for i in all_p_image_paths]
            all_text_paths = [str(i).replace("image", "text").replace(".jpg", ".txt") for i in all_p_image_paths]

            assert len(all_image_paths) == len(all_char_bbox_paths)
            assert len(all_image_paths) == len(all_text_paths)

            path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_char_bbox_paths, all_text_paths)).\
                shuffle(buffer_size=self.cfg['train_shuffle_buffer_size'], seed=self.cfg['train_shuffle_seed'])

            dataset = path_ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            if self.is_augment:
                dataset = dataset.map(self.data_augment(), num_parallel_calls=tf.data.AUTOTUNE)

        return dataset
