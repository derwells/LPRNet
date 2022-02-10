import os
import cv2
import random
import glob

CURRENT_DIR = os.path.abspath("")
DIRTY_PATH = os.path.join(CURRENT_DIR, "../dirty_data/ccpd_tiny")
CLEAN_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "clean_data/ccpd_tiny"))
AREA, TILT, BBOX, VERT, LPN, BRIGHT, BLUR = [_ for _ in range(7)]
PROVINCES = [
    "皖",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "京",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "警",
    "学",
    "O",
]
ALPHABETS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "O",
]
ADS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "O",
]

CHARS = [
    "京",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "皖",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "_",
]
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


def load_images_from_folder(folder, n_images=None):
    fnames = [os.path.join(folder, fname) for fname in os.listdir(folder)]
    random.shuffle(fnames)
    if n_images:
        fnames = fnames[:n_images]

    return fnames


def coords_to_int(arr):
    arr = arr.split("_")
    return [list(map(int, e.split("&"))) for e in arr]


def split_fname(fname):
    no_extension, ext = os.path.splitext(fname)
    fname_split = no_extension.split("-")
    fname_split[TILT] = coords_to_int(fname_split[TILT])
    fname_split[BBOX] = coords_to_int(fname_split[BBOX])
    fname_split[VERT] = coords_to_int(fname_split[VERT])

    return fname_split, ext


def reindex_lpn(lpn):
    lpn = lpn.split("_")
    lpn = [int(l) for l in lpn]

    new_lpn = []
    new_lpn.append(CHARS_DICT[PROVINCES[lpn[0]]])
    new_lpn.append(CHARS_DICT[ALPHABETS[lpn[1]]])
    for l in lpn[2:]:
        new_lpn.append(CHARS_DICT[ADS[l]])
    new_lpn = list(map(str, new_lpn))

    return "_".join(new_lpn)


if __name__ == "__main__":
    fnames = load_images_from_folder(DIRTY_PATH, 1000)

    for i, fname in enumerate(fnames):
        img = cv2.imread(fname)

        fname_split, ext = split_fname(fname)
        xs = [e[0] for e in fname_split[VERT]]
        ys = [e[1] for e in fname_split[VERT]]

        x_tl, x_br = min(xs), max(xs)
        y_tl, y_br = min(ys), max(ys)

        if not os.path.exists(CLEAN_PATH):
            os.makedirs(CLEAN_PATH)

        cropped_img = img[y_tl : y_br + 1, x_tl : x_br + 1].copy()
        new_lpn = reindex_lpn(fname_split[LPN])
        fname = f"{i}-{new_lpn}{ext}"
        fpath = os.path.join(CLEAN_PATH, fname)
        print(f"Writing {fpath}")
        cv2.imwrite(fpath, cropped_img)
