import os
import cv2
import random
import glob
import enum

import config
import constants

from classes.license_plate import *


def reindex(lpn_char):
    reindexed = constants.CHARS_DICT[lpn_char]
    reindexed = str(reindexed)

    return reindexed

def reindex_lpn(lpn):
    lpn = lpn.split("_")
    lpn = map(int, lpn)
    lpn = list(lpn)
    province_symbol = lpn[0]
    province_char = lpn[1]
    identifier = lpn[constants.LPN_SPLIT:]

    new_lpn = []

    # Province
    lpn_char = constants.PROVINCES[province_symbol]
    reindexed = reindex(lpn_char)
    new_lpn.append(reindexed)
    lpn_char = constants.ALPHABETS[province_char]
    reindexed = reindex(lpn_char)
    new_lpn.append(reindexed)

    # Rest of LPN
    for lpn_idx in identifier:
        lpn_char = constants.ADS[lpn_idx]
        reindexed = reindex(lpn_char)
        new_lpn.append(reindexed)
    print([constants.CHARS[int(i)] for i in new_lpn])

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

def clean_split_directory(target_dir, fnames):
    for idx, fname in enumerate(fnames):
        print(f"Processing {fname}")

        raw_lpn = RawLicensePlate(fname)

        new_lpn = reindex_lpn(raw_lpn.lpn)
        new_fname = f"{idx}-{new_lpn}{raw_lpn.extension}"
        new_fpath = os.path.join(target_dir, new_fname)

        fname_absolute_path = os.path.join(
            config.CCPD_PATH,
            fname
        )
        cropped_img = crop_image(
            fname_absolute_path,
            raw_lpn.vert
        )

        print(f"Writing {new_fpath}")
        cv2.imwrite(new_fpath, cropped_img)


if __name__ == "__main__":
    target_files = config.RAW_SPLIT_PATHS

    for key, target_file in target_files.items():
        target_dir = os.path.join(
            config.DATA_PATH,
            key
        )
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        fnames = get_target_fnames(target_file)
        clean_split_directory(target_dir, fnames)
