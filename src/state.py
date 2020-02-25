import copy
import cv2
import numpy as np


class State:
    MOVE_RANGE = 3

    @staticmethod
    def update(image_batch: np.ndarray,
               actions_batch: np.ndarray,
               in_place: bool = False) -> np.ndarray:
        """
        :param image_batch:
        :param actions_batch: action img.shape-d mx with values 0-(N_ACTIONS-1)
        :param in_place: if True, modifies the image_batch array, otherwise returns a new copy
        :return: the modified image batch
        """
        image_batch = image_batch.astype(np.float32)
        if in_place:
            image_batch_copy = image_batch
        else:
            image_batch_copy = copy.deepcopy(image_batch)
        neutral = (State.MOVE_RANGE - 1) / 2
        move = actions_batch.astype(np.float32)
        move = (move - neutral) / 255
        moved_image = image_batch_copy + move[:, np.newaxis, :, :]
        gaussian = np.zeros(image_batch_copy.shape, image_batch_copy.dtype)
        gaussian2 = np.zeros(image_batch_copy.shape, image_batch_copy.dtype)
        bilateral = np.zeros(image_batch_copy.shape, image_batch_copy.dtype)
        bilateral2 = np.zeros(image_batch_copy.shape, image_batch_copy.dtype)
        median = np.zeros(image_batch_copy.shape, image_batch_copy.dtype)
        box = np.zeros(image_batch_copy.shape, image_batch_copy.dtype)
        b, c, h, w = image_batch_copy.shape
        for i in range(0, b):
            if np.sum(actions_batch[i] == State.MOVE_RANGE) > 0:
                gaussian[i, 0] = cv2.GaussianBlur(image_batch_copy[i, 0], ksize=(5, 5), sigmaX=0.5)
            if np.sum(actions_batch[i] == State.MOVE_RANGE + 1) > 0:
                bilateral[i, 0] = cv2.bilateralFilter(image_batch_copy[i, 0], d=5, sigmaColor=0.1, sigmaSpace=5)
            if np.sum(actions_batch[i] == State.MOVE_RANGE + 2) > 0:
                median[i, 0] = cv2.medianBlur(image_batch_copy[i, 0], ksize=5)
            if np.sum(actions_batch[i] == State.MOVE_RANGE + 3) > 0:
                gaussian2[i, 0] = cv2.GaussianBlur(image_batch_copy[i, 0], ksize=(5, 5), sigmaX=1.5)
            if np.sum(actions_batch[i] == State.MOVE_RANGE + 4) > 0:
                bilateral2[i, 0] = cv2.bilateralFilter(image_batch_copy[i, 0], d=5, sigmaColor=1.0, sigmaSpace=5)
            if np.sum(actions_batch[i] == State.MOVE_RANGE + 5) > 0:
                box[i, 0] = cv2.boxFilter(image_batch_copy[i, 0], ddepth=-1, ksize=(5, 5))

        image_batch_copy = moved_image
        image_batch_copy = np.where(actions_batch[:, np.newaxis, :, :] == State.MOVE_RANGE, gaussian,
                                    image_batch_copy)
        image_batch_copy = np.where(actions_batch[:, np.newaxis, :, :] == State.MOVE_RANGE + 1, bilateral,
                                    image_batch_copy)
        image_batch_copy = np.where(actions_batch[:, np.newaxis, :, :] == State.MOVE_RANGE + 2, median,
                                    image_batch_copy)
        image_batch_copy = np.where(actions_batch[:, np.newaxis, :, :] == State.MOVE_RANGE + 3, gaussian2,
                                    image_batch_copy)
        image_batch_copy = np.where(actions_batch[:, np.newaxis, :, :] == State.MOVE_RANGE + 4, bilateral2,
                                    image_batch_copy)
        image_batch_copy = np.where(actions_batch[:, np.newaxis, :, :] == State.MOVE_RANGE + 5, box,
                                    image_batch_copy)

        return image_batch_copy
