import base64
import os.path
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Sequential
from keras.layers import RandomFlip, RandomZoom, RandomRotation, RandomTranslation
from matplotlib import pyplot as plt

from src.config import Config


class DataUtils:
    """
    Utility class for data preprocessing and augmentation tasks.

    Provides methods for encoding/decoding images, preprocessing, augmentation,
    and normalization of images.
    """

    @staticmethod
    def decode_base64_to_image(image_base64: str) -> np.ndarray:
        """
        Decode a base64-encoded image to a numpy array.

        Parameters:
        cxr_base64 (str): The base64-encoded image string.

        Returns:
        np.ndarray: The decoded image as a numpy array.
        """
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert('L')
        return np.array(image)

    @staticmethod
    def encode_image_to_base64(image: np.ndarray) -> str:
        """
        Encode a numpy array image to a base64 string.

        Parameters:
        image (np.ndarray): The image to encode.

        Returns:
        str: The base64-encoded image string.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, encoded_image = cv2.imencode(".png", image)
        encoded_image = base64.b64encode(encoded_image).decode("utf-8")
        return encoded_image

    @staticmethod
    def preprocess_image(image: np.ndarray or Image.Image, transform: Optional[cv2.CLAHE] = None) -> np.ndarray:
        """
        Preprocess an image by applying Gaussian blur, CLAHE, and resizing.

        Parameters:
        image (np.ndarray): The image to preprocess.
        transform (Optional[cv2.CLAHE]): The CLAHE transform to apply.

        Returns:
        np.ndarray: The preprocessed image.
        """
        image = np.array(image)
        if transform:
            image = transform.apply(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if not Config.IS_BASE_MODEL_PREPROCESS:
            # image = DataUtils.min_max_normalize_pos(image)
            image = image / 255.0
        image = cv2.resize(image, Config.IMAGE_SIZE)
        return image

    @staticmethod
    def augment_image(image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to an image.

        Parameters:
        image (np.ndarray): The image to augment.

        Returns:
        np.ndarray: The augmented image.
        """
        data_augmentation = Sequential(
            name='aug',
            layers=[
                RandomFlip('horizontal', name='hflip'),
                RandomFlip('vertical', name='vflip'),
                RandomRotation(0.05, fill_mode='constant', name='rot'),
                RandomZoom(height_factor=(-0.05, 0.05), fill_mode='constant', name='zoom'),
                RandomTranslation(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05), fill_mode='constant', name='translate'),
            ])
        return data_augmentation(image)

    @staticmethod
    def min_max_normalize_pos(image: np.ndarray) -> np.ndarray:
        """
        Normalize an image to the range [0, 1].

        Parameters:
        image (np.ndarray): The image to normalize.

        Returns:
        np.ndarray: The normalized image.
        """
        return (image - image.min()) / (image.max() - image.min())

    @staticmethod
    def min_max_normalize_sym(image: np.ndarray) -> np.ndarray:
        """
        Normalize an image to the range [-1, 1].

        Parameters:
        image (np.ndarray): The image to normalize.

        Returns:
        np.ndarray: The normalized image.
        """
        return ((image - image.min()) / ((image.max() - image.min()) / 2)) - 1.0

    @staticmethod
    def standardize_image(image: np.ndarray) -> np.ndarray:
        """
        Standardize an image to have a mean of 0 and a variance of 1.

        Parameters:
        image (np.ndarray): The image to standardize.

        Returns:
        np.ndarray: The standardized image.
        """
        return (image - image.mean()) / image.std()

    @staticmethod
    def inspect_tf_dataset(dataset: tf.data.Dataset, num_images: int = 5, save_dir: Optional[str] = None):
        """
        Visualize images from the dataset after preprocessing and augmentation.

        Parameters:
        dataset (tf.data.Dataset): The dataset to visualize.
        num_images (int): The number of images to visualize.
        """
        if not save_dir:
            save_dir = os.path.join(Config.LOG_DIR, 'sample_inspections')
            os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(30, 10))
        for images, labels in dataset.take(1):
            for i in range(num_images):
                ax = plt.subplot(1, num_images, i + 1)
                plt.imshow(images[i].numpy())
                plt.title(f'GT: {labels[i].numpy()}')
                plt.axis('off')
                plt.suptitle(f'Labels: {Config.PATHOLOGY_LABELS}')
                plt.tight_layout()
        plt.savefig(f'{os.path.join(save_dir, Config.START_TIME)}.png')
