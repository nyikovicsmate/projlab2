import time
from logging_config import logger
from image_prepocessor import ImagePrepocessor


if __name__ == '__main__':
    logger.info(f"Started at: {time.time()}")
    ip = ImagePrepocessor()
    ip.prepocess(overwrite=True)
