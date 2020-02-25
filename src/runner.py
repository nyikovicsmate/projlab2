import argparse
import pathlib
from src import *
from pixelwise_a3c_network import PixelwiseA3CNetwork
from image_dataset_manager import ImageDatasetManager


class Runner:
    w, h = (70, 70)
    data_dir = pathlib.Path.joinpath(PROJECT_ROOT, "data")
    idm = ImageDatasetManager(dataset_src_dir=pathlib.Path.joinpath(data_dir, "raw"),
                              dataset_dst_dir=pathlib.Path.joinpath(data_dir, "preprocessed"),
                              dst_shape=(w, h),
                              split_ratio=0.9)
    idm.preprocess(overwrite=False)

    @staticmethod
    def train():
        batch_size = 64
        batch_generator = Runner.idm.train_batch_generator(batch_size, randomize=True)
        network = PixelwiseA3CNetwork(input_shape=(batch_size, Runner.w, Runner.h, 1))
        network.train(batch_generator=batch_generator,
                      episodes=1000,
                      resume_training=False)

    @staticmethod
    def predict():
        batch_size = 64
        batch_generator = Runner.idm.train_batch_generator(batch_size, randomize=False)
        network = PixelwiseA3CNetwork(input_shape=(batch_size, Runner.w, Runner.h, 1))
        network.predict(batch_generator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_const", const=True, help="Train flag.")
    parser.add_argument("-p", "--predict", action="store_const", const=True, help="Predict flag.")
    args = parser.parse_args()

    if args.train:
        Runner.train()
    if args.predict:
        Runner.predict()
    else:
        logger.info("No mode selected, exiting.")
