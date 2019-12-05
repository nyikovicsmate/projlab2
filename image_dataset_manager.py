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
        else:
            logger.info("Using existing images.")
            train_img_path_list = os.listdir(self._dsm.train_dir)
            for img_path in train_img_path_list:
                full_path = os.path.join(self._dsm.train_dir, img_path)
                self._train_img_path_list.append(full_path)
            test_img_path_list = os.listdir(self._dsm.test_dir)
            for img_path in test_img_path_list:
                full_path = os.path.join(self._dsm.test_dir, img_path)
                self._test_img_path_list.append(full_path)
            logger.info(f"Found {len(train_img_path_list)} train, {len(test_img_path_list)} test images.")
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
                img = cv2.imread(self._train_img_path_list[idx], cv2.IMREAD_GRAYSCALE) / 255
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
            # yield the original and noisy images in a channel_first format
            yield np.array(orig_img_list).astype(np.float32)[:, np.newaxis, :, :], \
                  np.array(noisy_img_list).astype(np.float32)[:, np.newaxis, :, :]

    @staticmethod
    def _augment_image(img: np.ndarray) -> np.ndarray:
        """
        Augments the image.
        :param img: the image to augment
        :return: augmented image
        """
        # TODO
        raise NotImplementedError()

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
