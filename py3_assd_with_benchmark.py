from typing import Tuple
from pathlib import Path
from random import choices
from time import perf_counter as pf

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scipy.signal import correlate2d

from skimage.io import imread
from skimage.morphology import square
from skimage.morphology import binary_erosion

import matplotlib.pyplot as plt

from medpy.metric import assd


SELEM = square(3)
N_CPU = cpu_count() - 2


def binary_perimeter(img: np.ndarray) -> float:
    return np.sum(np.logical_xor(img, binary_erosion(img, selem=SELEM)))


def assd_surrogate(img_1: np.ndarray, img_2: np.ndarray, mode: str = "average") -> float:
    if mode == "fast":
        img = np.logical_xor(img_1, img_2)
        area = np.sum(img)
        length = max(binary_perimeter(img), 1)

        return area / length

    area = np.sum(np.logical_xor(img_1, img_2))
    length_1 = max(binary_perimeter(img_1), 1)
    length_2 = max(binary_perimeter(img_2), 1)

    res_1 = area / length_1
    res_2 = area / length_2

    if mode == "average":
        return 0.5 * (res_1 + res_2)
    elif mode == "minimum":
        return min(res_1, res_2)
    else:
        msn = (
            f'Supported modes are "fast", "average" and "minimum", got '
            f"{mode} instead"
        )
        raise ValueError(msn)


def compare_metrics(
    path_1: Path,
    path_2: Path,
    mode: str,
) -> Tuple[int, int, float, float]:
    img_1 = imread(str(path_1), as_gray=True)
    img_2 = imread(str(path_2), as_gray=True)

    ti = pf()
    assd_sur = assd_surrogate(img_1, img_2, mode)
    dt_sur = pf() - ti

    ti = pf()
    assd_org = assd(img_1, img_2)
    dt_org = pf() - ti

    return assd_org, assd_sur, dt_org, dt_sur


def main():
    # dir_input = Path(r"/media/hope0hermes/Data_Storage/Code/ASSD_benchmark/images_cells")
    dir_input = Path(r"/media/hope0hermes/Data_Storage/Code/ASSD_benchmark/images_nuclei")
    img_paths = list(dir_input.glob("*.png"))
    flag_dev = False
    n_pairs = 1000
    mode = "average"

    res = []
    if flag_dev:
        for path_1, path_2 in zip(choices(img_paths, k=n_pairs), choices(img_paths, k=n_pairs)):
            buff = compare_metrics(path_1, path_2, "fast")
            exit(0)
    else:
        with ProcessPoolExecutor() as pool:
            for buff in pool.map(
                compare_metrics,
                choices(img_paths, k=n_pairs),
                choices(img_paths, k=n_pairs),
                [mode for _ in range(n_pairs)],
            ):
                res.append(buff)

    df = pd.DataFrame(res, columns=["assd_org", "assd_sur", "dt_org", "dt_sur"])
    df["dt_ratio"] = df["dt_sur"] / df["dt_org"]
    df["assd_diff"] = df["assd_org"] - df["assd_sur"]

    _, ax = plt.subplots(ncols=3)
    ax[0].scatter(df["assd_org"], df["assd_sur"], alpha=0.2, s=5.0)
    ax[1].boxplot(df["assd_diff"], showfliers=False)
    ax[2].boxplot(df["dt_ratio"], showfliers=False)
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()