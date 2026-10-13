"""Reproduce the exam's train/test split exactly

Course: exam Ex1, every session. "Solutions that don't use the train/test split
correctly get 0 points." The split rule changes every time. Seen so far:
  Feb 2025  220 images in consecutive blocks of 55 per character
            (DeeDee, Louie, Dexter, Tommy); first 40 of each block = train.
  Feb 2025b 200 images, blocks of 50 (Mom, Dexter, Dad, DeeDee); first 40 = train.
  Jun 2025  annotation file lists images in RANDOM order with the character
            name; first 45 lines OF EACH CHARACTER = train, next 10 = test.
  Feb 2026  annotation file without names; the character comes from the file
            number range; first 40 lines of each character in FILE ORDER = train.
  Feb 2026b 001-150 faces train, 151-200 faces test; 201-215 / 216-220 are
            backgrounds: cut 10 random 40x40 patches from each as non-faces.

TASK
  split_blocks(n_images, block, n_train, first=1) -> (train_ids, test_ids)
      ids are ints (image numbers). Consecutive blocks of `block` images; the
      first n_train of each block go to train, the rest to test.
  split_by_order(names_in_order, labels_in_order, n_train, n_test) -> (train, test)
      Lists of file names. Walk the file in order; the first n_train files of
      each label go to train, the next n_test to test, anything after is unused.
  label_from_ranges(file_name, ranges) -> label
      ranges: list of (first_id, last_id, label), inclusive. "0057.jpg" -> 57.
  random_patches(img, n, size, rng) -> list of n size x size crops, each fully
      inside the image. rng is a np.random.Generator: draw x0 with
      rng.integers(0, W - size + 1) then y0 with rng.integers(0, H - size + 1),
      in that order, for each patch (so results are reproducible).
"""


def split_blocks(n_images, block, n_train, first=1):
    raise NotImplementedError


def split_by_order(names_in_order, labels_in_order, n_train, n_test):
    raise NotImplementedError


def label_from_ranges(file_name, ranges):
    raise NotImplementedError


def random_patches(img, n, size, rng):
    raise NotImplementedError
