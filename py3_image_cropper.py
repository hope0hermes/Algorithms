from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List
from shutil import rmtree

import numpy as np

from skimage.io import imsave
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import remove_small_holes

import matplotlib.pyplot as plt

import cv2

import rasterio


def read_image_objects(img_path: Path) -> List[np.ndarray]:
    with rasterio.open(img_path) as src:
        img_outline = src.read(1)
    img_label = label(remove_small_holes(img_outline, int(1e2)))
    return [255 * x.image.astype("uint16") for x in regionprops(img_label)]


def get_max_bbox_size(data_dir: Path) -> Tuple[int]:
    max_rows = 0
    max_cols = 0
    for img_path in data_dir.glob("*.[Ttp][Iin][Ffg]"):
        objects = read_image_objects(img_path)
        for obj in objects:
            max_rows = max(max_rows, obj.shape[0])
            max_cols = max(max_cols, obj.shape[1])
    return max_rows, max_cols


def get_paddings(img: np.ndarray, shape: Tuple[int]) -> Tuple[int]:
    bottom = 0
    top = 0
    left = 0
    right = 0
    if img.shape[0] < shape[0]:
        h_diff = shape[0] - img.shape[0]
        bottom = h_diff // 2
        top = h_diff - bottom
    if img.shape[1] < shape[1]:
        w_diff = shape[1] - img.shape[1]
        left = w_diff // 2
        right = w_diff - left
    return bottom, top, left, right


def pad_image(img: np.ndarray, shape: Tuple[int]) -> np.ndarray:
    bottom, top, left, right = get_paddings(img, shape)
    img_pad = cv2.copyMakeBorder(
        img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0
    )
    return img_pad


def write_image(img: np.ndarray, outfile: Path):
    imsave(str(outfile), img)


def main():
    dir_src = Path(r"/media/hope0hermes/Data_Storage/Datasets/bbbc/BBBC020_v1_outlines_nuclei")
    dir_dst = Path(r"/media/hope0hermes/Data_Storage/Code/ASSD_benchmark/images_nuclei")

    rmtree(dir_dst, ignore_errors=True)
    dir_dst.mkdir(exist_ok=True)

    shape = get_max_bbox_size(dir_src)
    print(shape)

    cnt = 0
    for img_path in dir_src.glob("*.[Ttp][Iin][Ffg]"):
        for obj in read_image_objects(img_path):
            cnt += 1
            img = pad_image(obj, shape)
            write_image(img, dir_dst / f"Fig_{cnt}.png")


if __name__ == "__main__":
    main()