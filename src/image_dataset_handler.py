import pathlib
from abc import ABC, abstractmethod
from typing import List


class ImageDatasetHandler(ABC):
    @abstractmethod
    def get_img_path_list(self,
                          dataset_src_dir: pathlib.Path) -> List[pathlib.Path]:
        """
        Returns the relative path to each image in the dataset.
        The returned paths should be relative to the dataset's base folder.
        :param dataset_src_dir: dataset base folder
        :return: list of absolute image paths
        """
        raise NotImplementedError()


class BSD68GrayHandler(ImageDatasetHandler):
    def get_img_path_list(self,
                          dataset_src_dir: pathlib.Path) -> List[pathlib.Path]:
        dataset_dir = pathlib.Path.joinpath(dataset_src_dir, "BSD68", "gray")
        train_dir = pathlib.Path.joinpath(dataset_dir, "train")
        test_dir = pathlib.Path.joinpath(dataset_dir, "test")
        img_path_list = [file for file in train_dir.iterdir()] + [file for file in test_dir.iterdir()]
        return img_path_list


class WaterlooExploartionHandler(ImageDatasetHandler):
    def get_img_path_list(self,
                          dataset_src_dir: pathlib.Path) -> List[pathlib.Path]:
        dataset_dir = pathlib.Path.joinpath(dataset_src_dir, "exploration_database_and_code", "pristine_images")
        img_path_list = [file for file in dataset_dir.iterdir()]
        return img_path_list
