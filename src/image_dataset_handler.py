import os
from abc import ABC, abstractmethod
from typing import List


class ImageDatasetHandler(ABC):
    @abstractmethod
    def get_img_path_list(self,
                          dataset_src_dir: str) -> List[str]:
        """
        Returns the relative path to each image in the dataset.
        The returned paths should be relative to the dataset's base folder.
        :param dataset_src_dir: dataset base folder
        :return: list of relative image path strings
        """
        raise NotImplementedError()


class BSD68GrayHandler(ImageDatasetHandler):
    def get_img_path_list(self,
                          dataset_src_dir: str) -> List[str]:
        dataset_dir = os.path.join(dataset_src_dir, "BSD68", "gray")
        train_dir = os.path.join(dataset_dir, "train")
        test_dir = os.path.join(dataset_dir, "test")
        img_path_list = []
        img_path_list.extend([os.path.join(train_dir, file) for file in os.listdir(train_dir)])
        img_path_list.extend([os.path.join(test_dir, file) for file in os.listdir(test_dir)])
        return img_path_list


class WaterlooExploartionHandler(ImageDatasetHandler):
    def get_img_path_list(self,
                          dataset_src_dir: str) -> List[str]:
        dataset_dir = os.path.join(dataset_src_dir, "exploration_database_and_code", "pristine_images")
        img_path_list = []
        img_path_list.extend([os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir)])
        return img_path_list
