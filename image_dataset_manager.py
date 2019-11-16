from directory_structure_manager import *
from image_dataset_handler import *
from typing import Tuple, List
import numpy as np
import cv2


class ImageDatasetManager:
    _handlers: List[ImageDatasetHandler]

    def __init__(self,
                 dataset_src_dir: str,
                 dataset_dst_dir: str,
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
            BSD68GrayHandler()
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
            logger.info("Deleting obsolete directory structure.")
            self._dsm.delete_directory_structure()
        if not self._dsm.is_created:
            logger.info("Creating directory structure.")
            self._dsm.create_directory_structure()
            raw_train_img_path_list = []
            raw_test_img_path_list = []
            for handler in self._handlers:
                list_ = handler.get_img_path_list(self._dataset_src_dir)
                split_idx = int(self.split_ratio * len(list_))
                raw_train_img_path_list.extend(list_[:split_idx])
                raw_test_img_path_list.extend(list_[split_idx:])
            logger.info(f"Found {len(raw_train_img_path_list)} train, {len(raw_test_img_path_list)} test images.")
            for idx, raw_train_img_path in enumerate(raw_train_img_path_list):
                img = cv2.imread(raw_train_img_path)
                img = self._process_image(img, self.dst_shape)
                full_path = os.path.join(self._dsm.train_dir, self.idx_to_img_name(idx))
                cv2.imwrite(full_path, img)
                self._train_img_path_list.append(full_path)
            logger.info(f"Processed {len(self._train_img_path_list)} train images.")
            for idx, raw_test_img_path in enumerate(raw_test_img_path_list):
                img = cv2.imread(raw_test_img_path)
                img = self._process_image(img, self.dst_shape)
                full_path = os.path.join(self._dsm.test_dir, self.idx_to_img_name(idx))
                cv2.imwrite(full_path, img)
                self._test_img_path_list.append(full_path)
            logger.info(f"Processed {len(self._test_img_path_list)} train images.")
        logger.info(f"Done processing.")

    def load_train_batch(self,
                         batch_size: int,
                         from_index: int = 0,
                         randomize: bool = False,
                         augment: bool = False) -> np.ndarray:
        """
        Loads batch_size number of preprocessed train images.
        :param batch_size:
        :param from_index: if given, function returns train_images[i: i+batch_size] for sequential processing
        :param randomize: if True, returns random sequence of images
        :param augment: if True, function augments images before returning
        :return: list of train images (batch_size, 1, width, height)
        """
        if from_index != 0 and randomize:
            raise Exception("Randomized output can not be sequentially indexed.")
        if randomize:
            random_idx_list = np.random.randint(0, len(self._train_img_path_list), batch_size)
            images = [cv2.imread(self._train_img_path_list[r_idx], cv2.IMREAD_GRAYSCALE) for r_idx in random_idx_list]
        else:
            if from_index >= len(self._train_img_path_list):
                # raise AttributeError(f"from_index is out of range. "
                #                      f"Got {from_index}, valid range [0, {len(self._train_img_path_list)-1}].")
                images = []
            else:
                images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in
                          self._train_img_path_list[from_index: from_index + batch_size]]
        if augment:
            images = np.array([self._augment_image(img) for img in images])
            # conver it to channels_first format
            images = images[:, np.newaxis, :, :]
            # convert to channels_last format
            # images = images[:, :, :, np.newaxis]
        return images

    @staticmethod
    def _augment_image(img: np.ndarray) -> np.ndarray:
        """
        Augments the image.
        :param img: the image to augment
        :return: augmented image
        """
        # TODO
        return img

    @staticmethod
    def _process_image(img: np.ndarray,
                       dst_shape: Tuple[int, int]) -> np.ndarray:
        """
        :param img: raw image
        :param dst_shape: desired shape (width, height)
        :return: prerocessed image
        """
        img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)
        # TODO crop instead of just scale
        img = cv2.resize(src=img, dsize=dst_shape, interpolation=cv2.INTER_NEAREST)
        return img

    @staticmethod
    def idx_to_img_name(idx: int) -> str:
        """
        :param idx: the index to convert
        :return: the full image name based on the index
        """
        return f"{idx:06d}.png"
