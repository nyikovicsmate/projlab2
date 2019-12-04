import time
from logging_config import logger
from image_dataset_manager import ImageDatasetManager
from pixelwise_a3c_network import PixelwiseA3CNetwork


if __name__ == '__main__':
    logger.info(f"Started at: {time.time()}")
    idm = ImageDatasetManager(dataset_src_dir="raw_images", dataset_dst_dir="preprocessed_images")
    idm.preprocess(overwrite=False)
    batch_generator = idm.train_batch_generator(64)
    network = PixelwiseA3CNetwork(input_shape=(64, 70, 70, 1))
    network.train(batch_generator=batch_generator,
                  epochs=10000)
