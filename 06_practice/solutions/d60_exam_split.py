import os
from collections import Counter


def split_blocks(n_images, block, n_train, first=1):
    train, test = [], []
    for i in range(n_images):
        (train if i % block < n_train else test).append(first + i)
    return train, test


def split_by_order(names_in_order, labels_in_order, n_train, n_test):
    seen = Counter()
    train, test = [], []
    for name, lab in zip(names_in_order, labels_in_order):
        k = seen[lab]
        seen[lab] += 1
        if k < n_train:
            train.append(name)
        elif k < n_train + n_test:
            test.append(name)
    return train, test


def label_from_ranges(file_name, ranges):
    num = int(os.path.splitext(os.path.basename(file_name))[0])
    for lo, hi, lab in ranges:
        if lo <= num <= hi:
            return lab
    raise ValueError(f"{file_name} is in no range")


def random_patches(img, n, size, rng):
    H, W = img.shape[:2]
    out = []
    for _ in range(n):
        x0 = int(rng.integers(0, W - size + 1))
        y0 = int(rng.integers(0, H - size + 1))
        out.append(img[y0:y0 + size, x0:x0 + size])
    return out
