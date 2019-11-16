import numpy as np
import cv2


class State:
    def __init__(self,
                 size,
                 move_range):
        # TODO better solution for move_range magic
        self.image = np.zeros(size, dtype=np.float32)
        self.move_range = move_range

    def reset(self,
              x: np.ndarray,
              n: np.ndarray) -> None:
        """
        :param x: input image list (batch_size, channels, width, height)
        :param n: input noise list (batch_size, channels, width, height)
        :return: None
        """
        self.image = x + n

    def step(self,
             act: np.ndarray) -> None:
        """
        For each pixel in the image applies the action indicated by the act mx.
        :param act: action img.shape-d mx with values 0-(N_ACTIONS-1)
        :return: None
        """
        # TODO convert it channel_last format
        # move_range = 3
        neutral = (self.move_range - 1) / 2
        move = act.astype(np.float32)
        move = (move - neutral) / 255
        moved_image = self.image + move[:, np.newaxis, :, :]
        gaussian = np.zeros(self.image.shape, self.image.dtype)
        gaussian2 = np.zeros(self.image.shape, self.image.dtype)
        bilateral = np.zeros(self.image.shape, self.image.dtype)
        bilateral2 = np.zeros(self.image.shape, self.image.dtype)
        median = np.zeros(self.image.shape, self.image.dtype)
        box = np.zeros(self.image.shape, self.image.dtype)
        b, c, h, w = self.image.shape
        # if there is any other predicted action than altering the pixel by 1, apply it for the whole image
        temp = 0
        for i in range(0, b):
            if np.sum(act[i] == self.move_range) > 0:
                gaussian[i, 0] = cv2.GaussianBlur(self.image[i, 0], ksize=(5, 5), sigmaX=0.5)
                temp += 1
            if np.sum(act[i] == self.move_range + 1) > 0:
                bilateral[i, 0] = cv2.bilateralFilter(self.image[i, 0], d=5, sigmaColor=0.1, sigmaSpace=5)
                temp += 1
            if np.sum(act[i] == self.move_range + 2) > 0:
                median[i, 0] = cv2.medianBlur(self.image[i, 0], ksize=5)
                temp += 1
            if np.sum(act[i] == self.move_range + 3) > 0:
                gaussian2[i, 0] = cv2.GaussianBlur(self.image[i, 0], ksize=(5, 5), sigmaX=1.5)
                temp += 1
            if np.sum(act[i] == self.move_range + 4) > 0:
                bilateral2[i, 0] = cv2.bilateralFilter(self.image[i, 0], d=5, sigmaColor=1.0, sigmaSpace=5)
                temp += 1
            if np.sum(act[i] == self.move_range + 5) > 0:
                box[i, 0] = cv2.boxFilter(self.image[i, 0], ddepth=-1, ksize=(5, 5))
                temp += 1

        self.image = moved_image
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range, gaussian, self.image)
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range + 1, bilateral, self.image)
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range + 2, median, self.image)
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range + 3, gaussian2, self.image)
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range + 4, bilateral2, self.image)
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range + 5, box, self.image)
