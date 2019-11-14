import os
import time
from logging_config import logger


class DirectoryStructureManager:
    def __init__(self,
                 root_dir: str):
        """
        :param root_dir: the desired root directory name for the prepocessed images
        """
        self.root_dir = os.path.join(os.getcwd(), root_dir)
        self.train_dir = os.path.join(self.root_dir, "train")
        self.test_dir = os.path.join(self.root_dir, "test")
        self.is_created = False

    def create_directory_structure(self) -> None:
        """
        Creates the directory structure in which the preprocessed images can later be placed.
        :return: None
        """
        logger.info(f"Started creating directory structure under\n{self.root_dir}")
        self._create_dirs()
        self.is_created = True
        logger.info(f"Finished creating directory structure under\n{self.root_dir}")

    def delete_directory_structure(self) -> None:
        """
        Deletes the directory structure created by _create_dirs() function.
        WARNING: removes every file found under self.root_dir!
        :return: None
        """
        self._delete_dirs()
        self.is_created = False

    def _create_dirs(self) -> None:
        """
        :return: None
        """
        # create preprocessed root directory
        if os.path.exists(self.root_dir):
            if len(os.listdir(self.root_dir)) != 0:
                raise FileExistsError(f"Root directory {self.root_dir} is not empty. "
                                      f"Only an empty directory can be root.")
        else:
            os.mkdir(self.root_dir)
            time.sleep(1)  # stop for 1 sec for the os to be able to catch up
        # create train and test directories
        os.mkdir(self.train_dir)
        os.mkdir(self.test_dir)

    def _delete_dirs(self) -> None:
        """
        :return: None
        """
        if os.path.exists(self.root_dir):
            self._delete_dir(self.train_dir)
            self._delete_dir(self.test_dir)
            self._delete_dir(self.root_dir)

    @staticmethod
    def _delete_dir(path: str) -> None:
        """
        :param path: file path
        :return: None
        """
        if os.path.exists(path):
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))
            time.sleep(1)  # stop for 1 sec for the os to be able to catch up
            os.rmdir(path)
