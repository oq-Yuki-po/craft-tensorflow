import cv2
import numpy as np

from src.dataset.score_genetator import generate_affinity_score, generate_gaussian, generate_region_score
from tests import OUTPUT_PATH


def test_generate_region_score(load_synthtext):

    image, character_bbox, _ = load_synthtext

    region_score, normalized_region_score = generate_region_score(image.shape, character_bbox)

    region_heatmap_img = cv2.applyColorMap(region_score, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(region_heatmap_img, 0.5, image, 0.5, 0)
    cv2.imwrite(f'{OUTPUT_PATH}/region_heatmap.png', overlay_img)

    assert region_score.dtype == np.dtype('uint8')
    assert normalized_region_score.dtype == np.dtype('float64')


def test_generate_affinity_score(load_synthtext):

    image, character_bbox, characters_list = load_synthtext

    affinity_score, normalized_affinity_score = generate_affinity_score(image.shape, character_bbox, characters_list)

    affinity_heatmap_img = cv2.applyColorMap(affinity_score, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(affinity_heatmap_img, 0.5, image, 0.5, 0)
    cv2.imwrite(f'{OUTPUT_PATH}/affinity_heatmap.png', overlay_img)

    assert affinity_score.dtype == np.dtype('uint8')
    assert normalized_affinity_score.dtype == np.dtype('float64')

def test_generate_gaussian():

    gaussian_heatmap = generate_gaussian()

    base_img = np.zeros([40,40,3],dtype=np.uint8)
    base_img.fill(255)
    gaussian_heatmap_img = cv2.applyColorMap(gaussian_heatmap, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(gaussian_heatmap_img, 1.0, base_img, 0.0, 0)
    cv2.imwrite(f'{OUTPUT_PATH}/gaussian_heatmap.png', overlay_img)

    assert gaussian_heatmap.dtype == np.dtype('uint8')
