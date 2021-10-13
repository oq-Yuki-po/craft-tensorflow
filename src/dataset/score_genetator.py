from typing import List, Tuple

import cv2
import numpy as np


def four_point_transform(image, pts):
    max_x, max_y = np.max(pts[:, 0]).astype(np.int32), np.max(pts[:, 1]).astype(np.int32)

    dst = np.array([
        [0, 0],
        [image.shape[1] - 1, 0],
        [image.shape[1] - 1, image.shape[0] - 1],
        [0, image.shape[0] - 1]], dtype="float32")

    # 透視変換
    M = cv2.getPerspectiveTransform(dst, pts)
    warped = cv2.warpPerspective(image, M, (max_x, max_y))

    return warped


def generate_gaussian(sigma: int = 10, spread: int = 4) -> np.ndarray:
    """正規分布の生成

    Args:
        sigma (int, optional): 正規分布のパラメータ. Defaults to 10.
        spread (int, optional): 正規分布のパラメータ. Defaults to 4.

    Returns:
        np.ndarray: 正規分布
    """
    extent = int(spread * sigma)
    center = spread * sigma / 2
    gaussian_heatmap = np.zeros([extent, extent], dtype=np.float32)

    for i_ in range(extent):
        for j_ in range(extent):
            gaussian_heatmap[i_, j_] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                -1 / 2 * ((i_ - center - 0.5) ** 2 + (j_ - center - 0.5) ** 2) / (sigma ** 2))

    gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap) * 255).astype(np.uint8)
    return gaussian_heatmap


def add_character(image, bbox):
    """文字のヒートマップを取得するための処理
    """
    # 左上の座標を整数値に変換
    top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)

    # bboxの中に0以下が含まれていない、bboxのx座標が画像の幅を超えてないか、bboxのy座標が画像の高さを超えてないか
    if np.any(bbox < 0) or np.any(bbox[:, 0] > image.shape[1]) or np.any(bbox[:, 1] > image.shape[0]):
        return image

    # 左上の座標を(0, 0)に移動
    bbox -= top_left[None, :]

    transformed = four_point_transform(generate_gaussian().copy(), bbox.astype(np.float32))

    start_row = max(top_left[1], 0) - top_left[1]
    start_col = max(top_left[0], 0) - top_left[0]
    end_row = min(top_left[1] + transformed.shape[0], image.shape[0])
    end_col = min(top_left[0] + transformed.shape[1], image.shape[1])

    image[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += transformed[start_row:end_row - top_left[1],
                                                                                   start_col:end_col - top_left[0]]

    return image


def add_affinity(image, bbox_1, bbox_2):
    '''文字間のヒートマップを取得する処理'''
    center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
    tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
    bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
    tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
    br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

    affinity = np.array([tl, tr, br, bl])

    return add_character(image, affinity)


def generate_region_score(image_size: Tuple[int, int, int], character_bbox: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Region Score Mapの生成

    Args:
        image_size (Tuple[int, int, int]): 画像サイズ
        character_bbox (np.ndarray): 文字のバウディングボックス
                                     shape (文字数, bboxの頂点(左下から反時計回りの順番), 座標)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Region Score, 正規化されたRegion Score
    """
    height, width, _ = image_size

    # ヒートマップの基を生成
    region_score = np.zeros([height, width], dtype=np.uint8)

    # 文字単位でRegionScoreを生成
    for i in range(character_bbox.shape[0]):
        region_score = add_character(region_score, character_bbox[i])

    return region_score.copy(), region_score.copy() / 255


def generate_affinity_score(image_size: Tuple[int, int, int], character_bbox: np.ndarray, text: List[List[str]])\
        -> Tuple[np.ndarray, np.ndarray]:
    """Affinity Scoreの生成

    Args:
        image_size (Tuple[int, int, int]): 画像サイズ
        character_bbox (np.ndarray): 文字のバウディングボックス
                                     shape (文字数, bboxの頂点(左下から反時計回りの順番), 座標)
        text (List[List[str]]): 埋め込まれている文字列の配列

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Affinity Score, 正規化されたAffinity Score
    """
    height, width, _ = image_size

    affinity_score = np.zeros([height, width], dtype=np.uint8)

    total_letters = 0

    # 単語単位でAffinityScoreを生成
    for word in text:
        for _ in range(len(word) - 1):
            affinity_score = add_affinity(affinity_score,
                                          character_bbox[total_letters].copy(),
                                          character_bbox[total_letters + 1].copy())
            total_letters += 1
        total_letters += 1

    return affinity_score.copy(), affinity_score.copy() / 255
