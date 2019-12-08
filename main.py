import time
import argparse
from logging_config import logger
from image_dataset_manager import ImageDatasetManager
from pixelwise_a3c_network import PixelwiseA3CNetwork


def train():
    # logger.info(f"Started at: {time.time()}")
    w, h = (70, 70)
    idm = ImageDatasetManager(dataset_src_dir="raw_images",
                              dataset_dst_dir="preprocessed_images",
                              dst_shape=(w, h),
                              split_ratio=0.9)
    idm.preprocess(overwrite=False)
    batch_size = 2
    batch_generator = idm.train_batch_generator(batch_size)
    network = PixelwiseA3CNetwork(input_shape=(batch_size, w, h, 1))
    network.train(batch_generator=batch_generator,
                  epochs=30000,
                  resume_training=False)


def predict():
    w, h = (70, 70)
    idm = ImageDatasetManager(dataset_src_dir="raw_images",
                              dataset_dst_dir="preprocessed_images",
                              dst_shape=(w, h),
                              split_ratio=0.9)
    idm.preprocess(overwrite=False)
    batch_size = 20
    batch_generator = idm.train_batch_generator(batch_size)
    network = PixelwiseA3CNetwork(input_shape=(batch_size, w, h, 1))
    network.predict(batch_generator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_const", const=True, help="Train flag.")
    parser.add_argument("-p", "--predict", action="store_const", const=True, help="Predict flag.")
    args = parser.parse_args()

    if args.train:
        train()
    if args.predict:
        predict()
    else:
        logger.info("No mode selected, exiting.")
