import time
from logging_config import logger
from image_dataset_manager import ImageDatasetManager


if __name__ == '__main__':
    logger.info(f"Started at: {time.time()}")
    idm = ImageDatasetManager(dataset_src_dir="raw_images", dataset_dst_dir="preprocessed_images")
    idm.preprocess()
