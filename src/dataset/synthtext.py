import pathlib
from typing import List

import cv2
import h5py
import imgaug.augmenters as iaa
import numpy as np
import typer
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from src.util import load_yaml


def split_text(txt: List[np.bytes_]) -> List[List[str]]:
    """テキストを文字単位に分割

    Args:
        txt (List[np.bytes_]): テキスト文字列の配列

    Returns:
        List[List[str]]: テキスト文字列の文字単位に分割した配列
    """

    txt = [i.decode('utf-8') for i in txt]

    splitted_text = []

    for words in txt:
        for word in words.split('\n'):
            splitted_char = []
            for char in list(word):
                if char != ' ':
                    splitted_char.append(char)
            splitted_text.append(splitted_char)

    return splitted_text


def generate_imgaug_bbox(char_bbox):

    imgaug_bbox = []

    for bbox in char_bbox:
        top_left = np.array([np.min(bbox[:, 0]), np.max(bbox[:, 1])]).astype(np.int32)
        bottom_right = np.array([np.max(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
        imgaug_bbox.append(BoundingBox(x1=top_left[0], y1=top_left[1], x2=bottom_right[0], y2=bottom_right[1]))

    return imgaug_bbox


def convert_bbox(imgaug_bboxes):
    all_bbox = []

    for i in imgaug_bboxes.bounding_boxes:
        bbox = i.to_keypoints()
        all_bbox.append(np.array([
            [bbox[0].x, bbox[0].y],
            [bbox[1].x, bbox[1].y],
            [bbox[2].x, bbox[2].y],
            [bbox[3].x, bbox[3].y]
        ]))

    return np.stack(all_bbox)


def main(file_path: str = 'src/dataset/SynthText.h5', saved_path: str = 'src/dataset/synthtext'):

    cfg = load_yaml()

    pathlib.Path(saved_path).mkdir(exist_ok=True)
    pathlib.Path(f'{saved_path}/char_bbox').mkdir(exist_ok=True)
    pathlib.Path(f'{saved_path}/image').mkdir(exist_ok=True)
    pathlib.Path(f'{saved_path}/text').mkdir(exist_ok=True)

    with h5py.File(file_path, 'r') as db:

        dataset = sorted(db['data'].keys())

        for image_name in dataset:

            image = db['data'][image_name][...]
            char_bbox = db['data'][image_name].attrs['charBB']
            txt = db['data'][image_name].attrs['txt']

            splitted_text = split_text(txt)

            char_bbox = char_bbox.transpose(2, 1, 0)

            imgaug_bbox = generate_imgaug_bbox(char_bbox)

            bbs = BoundingBoxesOnImage(imgaug_bbox, shape=image.shape)

            image_height, image_width, _ = cfg['input_image_size']

            aug = iaa.Resize((image_height, image_width))

            aug_image, aug_bboxes = aug(image=image, bounding_boxes=bbs)

            aug_char_bbox = convert_bbox(aug_bboxes)

            aug_image = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)

            cv2.imwrite(f"{saved_path}/image/{image_name}", aug_image)
            np.save(f"{saved_path}/char_bbox/{image_name.replace('.jpg', '')}.npy", aug_char_bbox)
            with open(f"{saved_path}/text/{image_name.replace('.jpg', '')}.txt", 'w') as f:
                for x in splitted_text:
                    f.write(str(x) + "\n")


if __name__ == "__main__":
    typer.run(main)
