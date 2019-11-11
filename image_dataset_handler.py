import os
import cv2
from typing import Tuple, List
from logging_config import logger


class ImageDatasetHandler:
    img_src_path_list: List[str]

    def __init__(self,
                 src_dir: str,
                 dst_dir: str,
                 dst_shape: Tuple[int, int],  # (width, height)
                 split_ratio: float):
        """
        :param src_dir: source directory of the raw images
        :param dst_dir: destination directory of the prepocessed images
        :param dst_shape: the desired shape of an image after prepocessing (width, height)
        :param split_ratio: train/test image set ratio (range between [0,1])
        """
        self.img_src_path_list = []
        self._src_dir = src_dir
        self._dst_dir = dst_dir
        self._dst_shape = dst_shape
        self._split_ratio = split_ratio
        self._img_list = []

        self.set_img_src_path_list()

    def set_img_src_path_list(self) -> None:
        """
        Fills the self.img_src_path_list variable with path strings to each raw image. The paths should be relative to
        self.src_dir (default ./raw_images).
        :return:
        """
        raise NotImplementedError()

    def process_images(self) -> None:
        """
        Convert images to the right format for deep learning.
        :return:
        """
        self._load_images()
        for i in range(len(self._img_list)):
            self._img_list[i] = self._scale(self._img_list[i], self._dst_shape)
            self._img_list[i] = self._grayscale(self._img_list[i])
        logger.info(f"{self.__class__.__name__}: Processed {len(self._img_list)} images.")
        self._save_images()

    def _load_images(self) -> None:
        """
        Stores all images in self.images list as cv2.Image objects.
        :return:
        """
        for path in self.img_src_path_list:
            img = cv2.imread(path)
            if img is None:
                raise Exception(f"Could not read image from path: {path}")
            self._img_list.append(img)
        logger.info(f"{self.__class__.__name__}: Loaded {len(self._img_list)} images.")

    def _save_images(self) -> None:
        """
        Splits the stored images according to self.split_ratio paramter, writes the resulting
        datasets into train_dir/test_dir respectfully. Image naming scheme: {index:06d}.png
        :return:
        """
        train_dir = os.path.join(self._dst_dir, 'train')
        test_dir = os.path.join(self._dst_dir, 'test')
        assert os.path.exists(train_dir) and os.path.exists(test_dir)
        split_idx = int(len(self._img_list) * self._split_ratio)
        train_img_list = self._img_list[:split_idx]
        test_img_list = self._img_list[split_idx:]
        first_free_train_idx = self._find_last_index(train_dir) + 1
        first_free_test_idx = self._find_last_index(test_dir) + 1
        for idx, img in enumerate(train_img_list):
            filename = f"{(first_free_train_idx + idx):06d}.png"
            cv2.imwrite(os.path.join(train_dir, filename), img)
        logger.info(f"{self.__class__.__name__}: Saved {len(train_img_list)} train images "
                    f"starting with index {first_free_train_idx} into\n{train_dir}")
        for idx, img in enumerate(test_img_list):
            filename = f"{(first_free_test_idx + idx):06d}.png"
            cv2.imwrite(os.path.join(test_dir, filename), img)
        logger.info(f"{self.__class__.__name__}: Saved {len(test_img_list)} test images "
                    f"starting with index {first_free_test_idx} into\n{test_dir}")

    @staticmethod
    def _find_last_index(path) -> int:
        files = os.listdir(path)
        max_idx = -1    # empty directory
        for f in files:
            filename, ext = f.split(".")  # filename is an index
            if int(filename) > max_idx:
                max_idx = int(filename)
        return max_idx

    @staticmethod
    def _scale(image,
               dst_shape):
        return cv2.resize(src=image, dsize=dst_shape, interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _grayscale(image):
        return cv2.cvtColor(src=image, code=cv2.COLOR_RGB2GRAY)


class BSD68GrayHandler(ImageDatasetHandler):
    def set_img_src_path_list(self) -> None:
        root_dir = os.path.join(self._src_dir, "BSD68", "gray")
        train_dir = os.path.join(root_dir, "train")
        test_dir = os.path.join(root_dir, "test")
        self.img_src_path_list.extend([os.path.join(train_dir, file) for file in os.listdir(train_dir)])
        self.img_src_path_list.extend([os.path.join(test_dir, file) for file in os.listdir(test_dir)])
