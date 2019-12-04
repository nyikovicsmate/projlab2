from typing import Tuple

import numpy as np
import cv2


class State:
    MOVE_RANGE = 3

    def __init__(self,
                 image_batch: np.ndarray):
        self.image_batch = image_batch.astype(np.float32)

    # def reset(self,
    #           x: np.ndarray) -> None:
    #     """
    #     :param x: input image list (batch_size, channels, width, height)
    #     :return: None
    #     """
    #     self.image_batch = x

    def step(self,
             actions: np.ndarray) -> None:
        """
        For each pixel in the image applies the action indicated by the act mx.
        :param actions: action (batch_size, channels, width, height) shaped mx with values 0-(N_ACTIONS-1)
        :return: None
        """
        # TODO convert it channel_last format
        # TODO better solution for move_range magic
        # move_range = 3
        neutral = (State.MOVE_RANGE - 1) / 2
        move = np.array(actions).astype(np.float32)
        # move = actions.astype(np.float32)
        move = (move - neutral) / 255
        moved_image = self.image_batch + move[:, np.newaxis, :, :]
        gaussian = np.zeros(self.image_batch.shape, self.image_batch.dtype)
        gaussian2 = np.zeros(self.image_batch.shape, self.image_batch.dtype)
        bilateral = np.zeros(self.image_batch.shape, self.image_batch.dtype)
        bilateral2 = np.zeros(self.image_batch.shape, self.image_batch.dtype)
        median = np.zeros(self.image_batch.shape, self.image_batch.dtype)
        box = np.zeros(self.image_batch.shape, self.image_batch.dtype)
        b, c, h, w = self.image_batch.shape
        # if there is any other predicted action than altering the pixel by 1, apply it for the whole image
        temp = 0
        for i in range(b):
            if np.sum(actions[i] == State.MOVE_RANGE) > 0:
                gaussian[i, 0] = cv2.GaussianBlur(self.image_batch[i, 0], ksize=(5, 5), sigmaX=0.5)
                temp += 1
            if np.sum(actions[i] == State.MOVE_RANGE + 1) > 0:
                bilateral[i, 0] = cv2.bilateralFilter(self.image_batch[i, 0], d=5, sigmaColor=0.1, sigmaSpace=5)
                temp += 1
            if np.sum(actions[i] == State.MOVE_RANGE + 2) > 0:
                median[i, 0] = cv2.medianBlur(self.image_batch[i, 0], ksize=5)
                temp += 1
            if np.sum(actions[i] == State.MOVE_RANGE + 3) > 0:
                gaussian2[i, 0] = cv2.GaussianBlur(self.image_batch[i, 0], ksize=(5, 5), sigmaX=1.5)
                temp += 1
            if np.sum(actions[i] == State.MOVE_RANGE + 4) > 0:
                bilateral2[i, 0] = cv2.bilateralFilter(self.image_batch[i, 0], d=5, sigmaColor=1.0, sigmaSpace=5)
                temp += 1
            if np.sum(actions[i] == State.MOVE_RANGE + 5) > 0:
                box[i, 0] = cv2.boxFilter(self.image_batch[i, 0], ddepth=-1, ksize=(5, 5))
                temp += 1

        self.image_batch = moved_image
        self.image_batch = np.where(actions[:, np.newaxis, :, :] == State.MOVE_RANGE, gaussian, self.image_batch)
        self.image_batch = np.where(actions[:, np.newaxis, :, :] == State.MOVE_RANGE + 1, bilateral, self.image_batch)
        self.image_batch = np.where(actions[:, np.newaxis, :, :] == State.MOVE_RANGE + 2, median, self.image_batch)
        self.image_batch = np.where(actions[:, np.newaxis, :, :] == State.MOVE_RANGE + 3, gaussian2, self.image_batch)
        self.image_batch = np.where(actions[:, np.newaxis, :, :] == State.MOVE_RANGE + 4, bilateral2, self.image_batch)
        self.image_batch = np.where(actions[:, np.newaxis, :, :] == State.MOVE_RANGE + 5, box, self.image_batch)
