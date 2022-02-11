import os
import cv2
import random
import glob
import enum

import constants
import config

CURRENT_DIR = os.path.abspath("")
DATA_DIR = os.path.join(
    CURRENT_DIR,
    "data"
)

CHARS_DICT = {
    char: i
    for i, char in enumerate(constants.CHARS)
}


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
        constants.TILT,
        constants.BBOX,
        constants.VERT,
    )
    for idx in make_coords_to_int:
        name_split[idx] = coords_to_int(name_split[idx])

    return name_split, extension

def reindex(lpn_char):
    reindexed = CHARS_DICT[lpn_char]
    reindexed = str(reindexed)

    return reindexed

def reindex_lpn(lpn):
    lpn = lpn.split("_")
    lpn = map(int, lpn)
    lpn = list(lpn)

    new_lpn = []

    # Province
    lpn_char = constants.PROVINCES[0]
    reindexed = reindex(lpn_char)
    new_lpn.append(reindexed)
    lpn_char = constants.ALPHABETS[1]
    reindexed = reindex(lpn_char)
    new_lpn.append(reindexed)

    # Rest of LPN
    others = lpn[constants.LPN_SPLIT:]
    for lpn_idx in others:
        lpn_char = constants.ADS[lpn_idx]
        reindexed = reindex(lpn_char)
        new_lpn.append(reindexed)

    fname_reindexed = "_".join(new_lpn)

    return fname_reindexed

def crop_image(fname, coords):
    img = cv2.imread(fname)

    xs = [
        c[0]
        for c in coords
    ]
    ys = [
        c[1]
        for c in coords
    ]

    x_tl, x_br = min(xs), max(xs)
    y_tl, y_br = min(ys), max(ys)

    cropped_img = img[y_tl:y_br+1, x_tl:x_br+1].copy()

    return cropped_img


def get_target_fnames(target_file):
    with open(target_file) as rf:
        lines = rf.readlines()
        cleaned = [line.rstrip('\n') for line in lines]
        return cleaned

if __name__ == "__main__":
    target_files = {
        "test": config.TEST_PATH,
        "val": config.VAL_PATH,
        "train": config.TRAIN_PATH
    }

    for key, target_file in target_files.items():
        target_dir = os.path.join(DATA_DIR, key)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        fnames = get_target_fnames(target_file)
        for i, fname in enumerate(fnames):
            print(f"Processing {fname}")

            fname_split, ext = split_fname(fname)
            new_lpn = reindex_lpn(fname_split[constants.LPN])
            new_fname = f"{i}-{new_lpn}{ext}"
            new_fpath = os.path.join(target_dir, new_fname)

            fname_abs = os.path.join(
                config.CCPD_PATH,
                fname
            )
            cropped_img = crop_image(
                fname_abs,
                fname_split[constants.VERT]
            )

            print(f"Writing {new_fpath}")
            cv2.imwrite(new_fpath, cropped_img)
