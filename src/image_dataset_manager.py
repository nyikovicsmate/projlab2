import numpy as np
import cv2
import pathlib
from typing import Tuple, List

from src import *
from directory_structure_manager import DirectoryStructureManager
from image_dataset_handler import ImageDatasetHandler, BSD68GrayHandler, WaterlooExploartionHandler


class ImageDatasetManager:
    _handlers: List[ImageDatasetHandler]

    def __init__(self,
                 dataset_src_dir: pathlib.Path,
                 dataset_dst_dir: pathlib.Path,
                 dst_shape: Tuple[int, int] = (70, 70),
                 split_ratio: float = 0.8,
                 ):
        """
        :param dataset_src_dir: source directory name of the raw images
        :param dataset_dst_dir: desired source directory name of the preprocessed images
        :param dst_shape: the desired shape of an image after prepocessing (width, height)
        :param split_ratio: train/test image set ratio (range between [0,1])
        """
        self.split_ratio = split_ratio
        self.dst_shape = dst_shape
        self._train_img_path_list = []
        self._test_img_path_list = []
        self._dataset_src_dir = dataset_src_dir
        self._dsm = DirectoryStructureManager(dataset_dst_dir)
        self._handlers = [
            BSD68GrayHandler(),
            WaterlooExploartionHandler()
        ]

    # noinspection DuplicatedCode
    def preprocess(self,
                   overwrite: bool = False) -> None:
        """
        Creates the directory structure for the preprocessed images, converts the raw images to the right format
        for deep learning and saves them in the created directories.
        :param overwrite: weather or not to overwrite existing directory structure
        :return: None
        """
        logger.info("Processing...")
        if overwrite:
            logger.info("Removing obsolete directory structure.")
            self._dsm.remove_directory_structure()
        if not self._dsm.is_created:
            logger.info("Creating directory structure.")
            self._dsm.create_directory_structure()
            raw_train_img_path_list = []
            raw_test_img_path_list = []
            for handler in self._handlers:
                try:
                    list_ = handler.get_img_path_list(self._dataset_src_dir)
                    split_idx = int(self.split_ratio * len(list_))
                    raw_train_img_path_list.extend(list_[:split_idx])
                    raw_test_img_path_list.extend(list_[split_idx:])
                except FileNotFoundError:
                    logger.warning(f"{handler.__class__.__name__} did not return any files, skipping.")
            logger.info(f"Found {len(raw_train_img_path_list)} train, {len(raw_test_img_path_list)} test images.")
            if len(raw_train_img_path_list) > 0:
                for idx, raw_train_img_path in enumerate(raw_train_img_path_list):
                    img = cv2.imread(str(raw_train_img_path))
                    img = self._process_image(img, self.dst_shape)
                    full_path = pathlib.Path.joinpath(self._dsm.train_dir, self.idx_to_img_name(idx))
                    cv2.imwrite(full_path, img)
                    self._train_img_path_list.append(full_path)
            logger.info(f"Processed {len(self._train_img_path_list)} train images.")
            if len(raw_test_img_path_list) > 0:
                for idx, raw_test_img_path in enumerate(raw_test_img_path_list):
                    img = cv2.imread(str(raw_test_img_path))
                    img = self._process_image(img, self.dst_shape)
                    full_path = pathlib.Path.joinpath(self._dsm.test_dir, self.idx_to_img_name(idx))
                    cv2.imwrite(full_path, img)
                    self._test_img_path_list.append(full_path)
            logger.info(f"Processed {len(self._test_img_path_list)} test images.")
        else:
            logger.info("Using existing images.")
            for img_path in self._dsm.train_dir.iterdir():
                self._train_img_path_list.append(img_path)
            for img_path in self._dsm.test_dir.iterdir():
                self._test_img_path_list.append(img_path)
            logger.info(f"Found {len(self._train_img_path_list)} train, {len(self._test_img_path_list)} test images.")
        # fail the preprocessing if no image has been processed
        if len(self._train_img_path_list) == 0 and len(self._test_img_path_list) == 0:
            self._dsm.remove_directory_structure()
            raise RuntimeError("No image has been processed, must be some kind of mistake.")
        logger.info(f"Done processing.")

    def train_batch_generator(self,
                              batch_size: int,
                              randomize: bool = False,
                              augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generator function, yields batch_size number of preprocessed train images.
        :param batch_size:
        :param randomize: if True, returns random sequence of images
        :param augment: if True, function augments images before returning
        :return: list of train images (batch_size, 1, width, height)
        """

        curr_start_index = 0
        # initialize a list with as many indexes as there are train images
        idx_list = np.arange(len(self._train_img_path_list))
        if randomize:
            # shuffle the indexes in-place
            np.random.shuffle(idx_list)
        while True:
            orig_img_list = []
            noisy_img_list = []
            curr_end_idx = curr_start_index + batch_size
            while curr_end_idx > len(idx_list):
                idx_list = np.append(idx_list, idx_list)
            for idx in idx_list[curr_start_index:curr_end_idx]:
                img = cv2.imread(str(self._train_img_path_list[idx]), cv2.IMREAD_GRAYSCALE) / 255
                if augment:
                    img = self._augment_image(img)
                orig_img_list.append(img)
                noisy_img_list.append(self._add_noise(img))
            # increase the starting index for the next generation
            curr_start_index += batch_size
            # if we have reached the end of the available images list,
            # re-initialize (and re-randomize if necessary) the index list and start from the beginning again
            if curr_start_index > len(self._train_img_path_list):
                curr_start_index = 0
                idx_list = np.arange(len(self._train_img_path_list))
                if randomize:
                    # shuffle the indexes in-place
                    np.random.shuffle(idx_list)
            orig_img_list = np.array(orig_img_list, dtype=np.float32)
            noisy_img_list = np.array(noisy_img_list, dtype=np.float32)
            # yield the original and noisy images in a channel_first format
            yield orig_img_list[:, np.newaxis, :, :], noisy_img_list[:, np.newaxis, :, :]

    @staticmethod
    def _augment_image(img: np.ndarray) -> np.ndarray:
        """
        Augments the image.
        :param img: the image to augment
        :return: augmented image
        """
        # TODO
        pass

    @staticmethod
    def _add_noise(img: np.ndarray) -> np.ndarray:
        """
        Adds noise to the grayscale image.
        :param img:
        :return: image with added noise
        """
        img = np.copy(img)
        # TODO better/more noises
        for i in range(img.shape[0]):
            if i % 2 == 0:
                img[i][0::2] = 1
            else:
                img[i][1::2] = 1
        return img

    @staticmethod
    def _process_image(img: np.ndarray,
                       dst_shape: Tuple[int, int]) -> np.ndarray:
        """
        Transforms parameter image into grayscale and clips an arbitrary sized section out of it.
        :param img: RGB image
        :param dst_shape: desired clip shape (width, height)
        :return: clipped grayscale image
        """
        dst_w, dst_h = dst_shape
        # resize original image before clipping if needed
        resize_factor_w = 0
        resize_factor_h = 0
        src_h, src_w, src_c = img.shape  # size format is (w,h) image_shape however is (h,w) (cv2 speciality)
        if dst_w > src_w:
            resize_factor_w = np.ceil(dst_w / src_w).astype(np.uint8)
        if dst_h > src_h:
            resize_factor_h = np.ceil(dst_h / src_h).astype(np.uint8)
        if resize_factor_w != 0 or resize_factor_h != 0:
            resize_factor = np.maximum(resize_factor_w, resize_factor_h)
            src_h *= resize_factor
            src_w *= resize_factor
            img = cv2.resize(src=img, dsize=(src_w, src_h), interpolation=cv2.INTER_LINEAR)
        # randomly clip image
        width_clip_idx_start = 0 if src_w == dst_w else np.random.randint(low=0, high=src_w - dst_w)
        height_clip_idx_start = 0 if src_w == dst_h else np.random.randint(low=0, high=src_h - dst_h)
        img = img[height_clip_idx_start: height_clip_idx_start + dst_h,
                width_clip_idx_start: width_clip_idx_start + dst_w]
        # grayscale image
        img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)
        return img

    @staticmethod
    def idx_to_img_name(idx: int) -> str:
        """
        :param idx: the index to convert
        :return: the full image name based on the index
        """
        return f"{idx:06d}.png"
