import pathlib
import time
from src import logger


class DirectoryStructureManager:
    def __init__(self,
                 root_dir: pathlib.Path):
        """
        :param root_dir: the desired root directory name for the prepocessed images
        """
        self.root_dir = root_dir
        self.train_dir = pathlib.Path.joinpath(self.root_dir, "train")
        self.test_dir = pathlib.Path.joinpath(self.root_dir, "test")
        self.is_created = True if pathlib.Path.exists(self.train_dir) and pathlib.Path.exists(self.test_dir) else False

    def create_directory_structure(self) -> None:
        """
        Creates the directory structure in which the preprocessed images can later be placed.
        :return: None
        """
        logger.info(f"Started creating directory structure under\n{self.root_dir}")
        self._create_dirs()
        self.is_created = True
        logger.info(f"Finished creating directory structure under\n{self.root_dir}")

    def remove_directory_structure(self) -> None:
        """
        Deletes the directory structure created by self.create_directory_structure() function.
        WARNING: removes every file found under self.root_dir!
        :return: None
        """
        self._remove_dirs()
        self.is_created = False

    def _create_dirs(self) -> None:
        """
        :return: None
        """
        # create preprocessed root directory
        if pathlib.Path.exists(self.root_dir):
            if pathlib.Path.stat(self.root_dir).st_size != 0:
                raise FileExistsError(f"Root directory {self.root_dir} is not empty. "
                                      f"Only an empty directory can be root.")
        else:
            logger.info(f"Creating {self.root_dir}")
            pathlib.Path.mkdir(self.root_dir)
            time.sleep(1)  # stop for 1 sec for the os to be able to catch up
        # create train and test directories
        logger.info(f"Creating {self.train_dir}")
        pathlib.Path.mkdir(self.train_dir)
        logger.info(f"Creating {self.test_dir}")
        pathlib.Path.mkdir(self.test_dir)

    def _remove_dirs(self) -> None:
        """
        :return: None
        """
        if pathlib.Path.exists(self.root_dir):
            self._remove_dir(self.train_dir)
            self._remove_dir(self.test_dir)
            self._remove_dir(self.root_dir)

    @staticmethod
    def _remove_dir(path: pathlib.Path) -> None:
        """
        :param path: directory path
        :return: None
        """
        logger.info(f"Removing {path}")
        for file in path.iterdir():
            pathlib.Path.unlink(file)
        time.sleep(1)  # stop for 1 sec for the os to be able to catch up
        pathlib.Path.rmdir(path)
