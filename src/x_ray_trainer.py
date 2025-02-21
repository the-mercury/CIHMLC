import logging
import os
from typing import Optional, List, Dict

import tensorflow as tf
from keras import Model as KerasModel
from pandas import DataFrame as PandasDataFrame

from src.config import Config
from src.data.chexpert_data_loader import CheXpertDataLoader
from src.helpers.model_builder import ModelBuilder

logger = logging.getLogger('XRayTrainer')


class XRayTrainer:
    """
    XRayTrainer class for training a model on the CheXpert dataset with hierarchical classification.

    This class handles data loading, model building, training, and saving the model.
    """

    def __init__(self,
                 parent_indices: List[int] = None, child_indices: List[List[int]] = None,
                 penalties: Optional[Dict[int, Dict[int, float]]] = None,
                 hierarchy_penalty: Optional[float] = None,
                 penalty_scale: Optional[float] = 1.0,
                 use_computed_penalties: Optional[bool] = False,
                 base_model: Optional[str] = None):
        """
        Initialize the XRayTrainer.

        Parameters:
        parent_indices (List[int]): List of parent indices.
        child_indices (List[List[int]]): List of child indices.
        penalties (Optional[Dict[int, Dict[int, float]]]): Penalties for the hierarchical loss function.
        hierarchy_penalty (Optional[float]): Fixed penalty for the hierarchical loss function.
        penalty_scale (Optional[float]): Scaling factor for the penalties.
        base_model (Optional[float]): Base model architecture.
        """
        logger.info('Creating a new XRayTrainer instance')
        logger.info(f'Num GPUs Available: {len(tf.config.list_physical_devices("GPU"))}')

        super().__init__()
        self.__data_loader = CheXpertDataLoader()
        self.__num_classes: int = self.__data_loader.get_num_classes()
        if parent_indices is None or child_indices is None:
            self.__parent_indices, self.__child_indices = self.__data_loader.get_parent_child_indices()
        else:
            self.__parent_indices = parent_indices
            self.__child_indices = child_indices
        if use_computed_penalties:
            self.__hierarchy_penalty = None
            if penalties is None:
                self.__penalties = self.__data_loader.get_penalties()
            else:
                self.__penalties = penalties
        else:
            self.__penalties = None
            self.__hierarchy_penalty = hierarchy_penalty
        self.__penalty_scale = penalty_scale
        self.__model_builder = ModelBuilder(
            parent_indices=self.__parent_indices,
            child_indices=self.__child_indices,
            penalties=self.__penalties,
            hierarchy_penalty=self.__hierarchy_penalty,
            penalty_scale=self.__penalty_scale,
            num_classes=self.__num_classes,
            base_model=base_model
        )
        self.__model = self.__model_builder.get_model()
        self.__model_config = self.__model_builder.get_config()
        logger.info(f'XRayTrainer initialized with model: {self.__model.name}')

    def get_data_loader(self) -> CheXpertDataLoader:
        """
        Get the data loader instance.

        Returns:
        DataLoader: A data loader instance for the dataset.
        """
        return self.__data_loader

    def get_model(self) -> KerasModel:
        """
        Get the compiled model.

        Returns:
        keras.Model: The compiled Keras model.
        """
        return self.__model

    def get_config(self) -> PandasDataFrame:
        """
        Get the model configuration.

        Returns:
        DataFrame: The model configuration.
        """
        return self.__model_config

    def save_model(self):
        """
        Save the trained model to the specified path.
        """
        model_save_id = f'{self.__model.name}.keras'
        self.__model.save(os.path.join(Config.NEW_MODEL_PATH, model_save_id))
        PandasDataFrame.from_dict(self.__penalties, orient='index').to_csv(
            os.path.join(Config.NEW_MODEL_PATH, f'{self.__model.name}_config.csv'))

        logger.info(f'Model saved in {os.path.join(Config.NEW_MODEL_PATH, model_save_id)}!')
        if self.__penalties:
            logger.info(f'Penalty configuration saved at: '
                        f'{os.path.join(Config.NEW_MODEL_PATH, f"{self.__model.name}_config.csv")}')
