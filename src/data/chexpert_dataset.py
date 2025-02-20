import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import Config

logger = logging.getLogger('CheXpertDataset')


class CheXpertDataset:
    """
    Data sequence class for CheXpert dataset using TensorFlow's tf.data API.
    """

    def __init__(self, df: pd.DataFrame, unique_labels: List[str], input_folders: List[str], batch_size: int,
                 shuffle: bool, preprocess: bool, augment: bool, preprocess_unit: Optional[callable]):
        """
        Initialize the data sequence.

        Parameters:
        df (pd.DataFrame): DataFrame containing the dataset information.
        unique_labels (List[str]): List of unique labels.
        input_folders (List[str]): Paths to the folders containing images.
        batch_size (int): Batch size for the data generator.
        shuffle (bool): Whether to shuffle the dataset.
        preprocess_unit (Optional[callable]): Preprocessing function for the base model.
        """
        self.__df = df
        self.__unique_labels = unique_labels
        self.__input_folders = input_folders
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__is_preprocess = preprocess
        self.__transform = cv2.createCLAHE(clipLimit=3, tileGridSize=(10, 10))
        self.__is_augment = augment
        self.__preprocess_unit = preprocess_unit
        self.__dataset = self.__create_dataset()

    @staticmethod
    def __load_image(path: str) -> tf.Tensor:
        """
        Load an image from the specified path.

        Parameters:
        path (str): Path to the image file.

        Returns:
        tf.Tensor: The loaded image tensor.
        """
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, Config.IMAGE_SIZE)
        return image

    def __apply_clahe(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

        Parameters:
        image (tf.Tensor): The image tensor to apply CLAHE.

        Returns:
        tf.Tensor: The image tensor after applying CLAHE.
        """
        image = tf.cast(image, tf.uint8)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.numpy_function(self.__apply_clahe_np, [image], tf.uint8)
        image.set_shape(Config.IMAGE_SIZE)
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)
        return image

    def __apply_clahe_np(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE using OpenCV.

        Parameters:
        image (np.ndarray): The grayscale image as a NumPy array.

        Returns:
        np.ndarray: The image after applying CLAHE.
        """
        image = self.__transform.apply(image)
        return image

    @staticmethod
    def __min_max_normalize(image: tf.Tensor) -> tf.Tensor:
        """
        Normalize the image tensor to the range [0, 1] using min-max normalization.

        Parameters:
        image (tf.Tensor): The image tensor to normalize.

        Returns:
        tf.Tensor: The normalized image tensor.
        """
        min_val = tf.reduce_min(image)
        max_val = tf.reduce_max(image)
        normalized_image = (image - min_val) / (max_val - min_val)
        return normalized_image

    def __preprocess(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocess the image.

        Parameters:
        image (tf.Tensor): The image tensor to preprocess.
        label (tf.Tensor): The corresponding label tensor.

        Returns:
        Tuple[tf.Tensor, tf.Tensor]: The preprocessed image and label tensors.
        """
        if self.__preprocess_unit:
            image = self.__preprocess_unit(image)
        # image = self.__apply_clahe(image)
        # image = self.__min_max_normalize(image)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
        return image, label

    @staticmethod
    def __augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply data augmentation to the image.

        Parameters:
        image (tf.Tensor): The image tensor to augment.
        label (tf.Tensor): The corresponding label tensor.

        Returns:
        Tuple[tf.Tensor, tf.Tensor]: The augmented image and label tensors.
        """
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, label

    # WARNING:tensorflow:AutoGraph could not transform <bound method CheXpertDataset.__load_and_preprocess_image of <src.data.chexpert_dataset.CheXpertDataset object at 0x7ff9b00ace50>> and will run it as-is.
    # Cause: mangled names are not yet supported
    # To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert

    ### >>> last checked on Sep30, 2024 <<< ###
    @tf.autograph.experimental.do_not_convert
    def __load_and_preprocess_image(self, path: str, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load and preprocess the image.

        Parameters:
        path (str): Path to the image file.
        label (tf.Tensor): The corresponding label tensor.

        Returns:
        Tuple[tf.Tensor, tf.Tensor]: The preprocessed image and label tensors.
        """
        image = self.__load_image(path)
        if self.__is_preprocess:
            image, label = self.__preprocess(image, label)
        if self.__is_augment:
            image, label = self.__augment(image, label)
        return image, label

    def __create_dataset(self) -> tf.data.Dataset:
        """
        Create the dataset using TensorFlow's tf.data API.

        Returns:
        tf.data.Dataset: The created dataset.
        """
        paths = self.__df['Path'].apply(self.__resolve_path).tolist()
        labels = self.__df[self.__unique_labels].values.tolist()

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(self.__load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

        if self.__shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(self.__batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def __resolve_path(self, relative_path: str) -> str:
        """
        Resolve the full path of a file from the list of input folders.

        Parameters:
        relative_path (str): The relative path of the file.

        Returns:
        str: The full path of the file if it exists in any of the input folders.

        Raises:
        FileNotFoundError: If the file does not exist in any of the provided directories.
        """
        for folder in self.__input_folders:
            full_path = os.path.join(folder, relative_path)
            if os.path.exists(full_path):
                return full_path
        raise logger.error(FileNotFoundError(f'File not found in any of the provided directories: {relative_path}'))

    def get_dataset(self) -> tf.data.Dataset:
        """
        Get the data sequence as a tf.data.Dataset.

        Returns:
        tf.data.Dataset: The data sequence dataset.
        """
        return self.__dataset
