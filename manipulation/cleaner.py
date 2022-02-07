import os
import cv2
import random
import glob

CURRENT_DIR = os.path.abspath('')
DIRTY_PATH = os.path.join(CURRENT_DIR, '../dirty_data/ccpd_val')
CLEAN_PATH = os.path.abspath(
    os.path.join(CURRENT_DIR, '..', 'clean_data/ccpd_val')
)
AREA, TILT, BBOX, VERT, LPN, BRIGHT, BLUR = [
    _ for _ in range(7)
]


def load_images_from_folder(folder, n_images=None):
    fnames = [
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
    ]
    random.shuffle(fnames)
    if n_images:
        fnames = fnames[:n_images]

    return fnames

def coords_to_int(arr):
    arr = arr.split('_')
    return [
        list(
            map(int, e.split('&'))
        )
        for e in arr
    ]

def split_fname(fname):
    no_extension, ext = os.path.splitext(fname)
    fname_split = no_extension.split('-')
    fname_split[TILT] = coords_to_int(fname_split[TILT])
    fname_split[BBOX] = coords_to_int(fname_split[BBOX])
    fname_split[VERT] = coords_to_int(fname_split[VERT])

    return fname_split, ext

if __name__ == '__main__':
    fnames = load_images_from_folder(DIRTY_PATH, 5000)

    for i, fname in enumerate(fnames):
        img = cv2.imread(fname)
    
        fname_split, ext = split_fname(fname)
        xs = [e[0] for e in fname_split[VERT]]
        ys = [e[1] for e in fname_split[VERT]]

        x_tl, x_br = min(xs), max(xs)
        y_tl, y_br = min(ys), max(ys)

        if not os.path.exists(CLEAN_PATH):
            os.makedirs(CLEAN_PATH)
        
        cropped_img = img[y_tl:y_br+1, x_tl:x_br+1].copy()
        fname = f'{i}-{fname_split[LPN]}.{ext}'
        fpath = os.path.join(CLEAN_PATH, fname)
        print(f"Writing {fpath}")
        cv2.imwrite(
            fpath,
            cropped_img
        )
