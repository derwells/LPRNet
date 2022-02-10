import os
import cv2
import random
import glob
import enum

import constants

from constants import ccpd_fname, ccpd_lpn

CURRENT_DIR = os.path.abspath("")
DATA_DIR = os.path.join(
    CURRENT_DIR, 
    "../data"
)
CLEAN_PATH = os.path.join(
    DATA_DIR, 
    "/data/clean"
)
DIRTY_PATH = os.path.join(
    DATA_DIR, 
    "/data/CCPD2019"
)

CHARS_DICT = {
    char: i 
    for i, char in enumerate(constants.CHARS)
}



def load_images_from_folder(folder, n_images=None):
    fnames = [
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
    ]
    random.shuffle(fnames)
    if n_images:
        fnames = fnames[:n_images]

    return fnames


def coords_to_int(coords):
    coords = coords.split("_")
    coords_int = []
    for c in coords:
        tmp = c.split("&")
        tmp = map(int, tmp)
        tmp = list(tmp)
        coords_int.append(tmp)

    return coords_int


def split_fname(fname):
    name, extension = os.path.splitext(fname)
    name_split = name.split("-")

    make_coords_to_int = (
        ccpd_fname.TILT,
        ccpd_fname.BBOX,
        ccpd_fname.VERT,
    )
    for idx in make_coords_to_int:
        name_split[idx] = coords_to_int(name_split[idx])

    return name_split, extension


def reindex_lpn(lpn):
    lpn = lpn.split("_")
    lpn = map(int, lpn)
    lpn = list(lpn)

    new_lpn = []

    # Province
    for idx in ccpd_lpn.PROVINCES:
        lpn_idx = lpn[idx]
        lpn_char = constants.ALPHABETS[lpn_idx]
        reindexed = CHARS_DICT[lpn_char]
        reindexed = str(reindexed)
        new_lpn.append(reindexed)

    # Rest of LPN
    others = lpn[ccpd_lpn.OTHERS:]
    for lpn_idx in others:
        lpn_char = constants.ADS[lpn_idx]
        reindexed = CHARS_DICT[lpn_char]
        reindexed = str(reindexed)
        new_lpn.append(reindexed)

    fname_reindexed = "_".join(new_lpn)

    return fname_reindexed


if __name__ == "__main__":
    fnames = load_images_from_folder(DIRTY_PATH)

    for i, fname in enumerate(fnames):
        img = cv2.imread(fname)

        fname_split, ext = split_fname(fname)
        xs = [e[0] for e in fname_split[ccpd_fname.VERT]]
        ys = [e[1] for e in fname_split[ccpd_fname.VERT]]

        x_tl, x_br = min(xs), max(xs)
        y_tl, y_br = min(ys), max(ys)

        if not os.path.exists(CLEAN_PATH):
            os.makedirs(CLEAN_PATH)

        cropped_img = img[y_tl : y_br + 1, x_tl : x_br + 1].copy()
        new_lpn = reindex_lpn(fname_split[ccpd_fname.LPN])
        fname = f"{i}-{new_lpn}{ext}"
        fpath = os.path.join(CLEAN_PATH, fname)
        print(f"Writing {fpath}")
        cv2.imwrite(fpath, cropped_img)
