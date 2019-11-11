import os
import time
from typing import List

from image_dataset_handler import *
from logging_config import logger


class ImagePrepocessor:
    _handlers: List[ImageDatasetHandler]
    IMG_SHAPE = (70, 70)
    SPLIT_RATIO = 0.8

    def __init__(self):
        self._raw_img_dir = "raw_images"
        self._prepocessed_img_dir = "prepocessed_images"
        self._handlers = [
            BSD68GrayHandler(self._raw_img_dir, self._prepocessed_img_dir, self.IMG_SHAPE, self.SPLIT_RATIO)
        ]

    def prepocess(self,
                  overwrite: bool = False) -> None:
        logger.info("Processing...")
        if overwrite:
            logger.info("Creating directory structure.")
            if self._delete_dirs():
                self._create_dirs()
        else:
            logger.info("Using existing directory structure.")
        # TODO: multithreading
        for h in self._handlers:
            h.process_images()
        logger.info("Done preprocessing.")

    def _create_dirs(self) -> None:
        # create preprocessed root directory
        root_dir = os.path.join(os.getcwd(), self._prepocessed_img_dir)
        if os.path.exists(root_dir):
            logger.info(f"{root_dir} already exists, did you maybe forget to call delete_dirs()?")
        else:
            os.mkdir(root_dir)
            time.sleep(1)  # stop for 1 sec for the os to be able to catch up
        # create train and test directories
        train_dir = os.path.join(root_dir, "train")
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
            time.sleep(1)  # stop for 1 sec for the os to be able to catch up
        test_dir = os.path.join(root_dir, "test")
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
            time.sleep(1)  # stop for 1 sec for the os to be able to catch up

    def _delete_dirs(self) -> bool:
        root_dir = os.path.join(os.getcwd(), self._prepocessed_img_dir)
        train_dir = os.path.join(root_dir, "train")
        test_dir = os.path.join(root_dir, "test")
        self._delete_dir(train_dir)
        self._delete_dir(test_dir)
        if os.path.exists(root_dir):
            list_ = os.listdir(root_dir)
            if len(list_) > 0:
                logger.warning("Stopping deletion. Found misplaced files:\n{filelist}"
                               .format(filelist="\n".join([str(f) for f in list_])))
                return False
            time.sleep(1)  # stop for 1 sec for the os to be able to catch up
            os.rmdir(root_dir)
        time.sleep(1)  # stop for 1 sec for the os to be able to catch up
        return True

    @staticmethod
    def _delete_dir(path):
        if os.path.exists(path):
            for file in os.listdir(path):
                try:
                    os.remove(os.path.join(path, file))
                except IsADirectoryError:
                    logger.warning(f"Stopping deletion. Found misplaced directory:\n{file}")
                    return False
            time.sleep(1)  # stop for 1 sec for the os to be able to catch up
            os.rmdir(path)
