import logging
import os
from typing import List, Tuple, Dict

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.config import Config
from src.data.chexpert_dataset import CheXpertDataset
from src.helpers.model_utils import ModelUtils

logger = logging.getLogger('CheXpertDataLoader')


class CheXpertDataLoader:
    """
    DataLoader class for loading and preprocessing CheXpert data.

    This class handles data loading, preprocessing, and generating data sequences
    for training, validation, and testing.
    """

    def __init__(self):
        """
        Initialize the DataLoader.
        """
        logger.info('Creating a new CheXpertDataLoader instance')
        super().__init__()
        self.__is_uncertain_label: bool = Config.IS_UNCERTAIN_LABEL
        self.__uncertain_label: str = Config.UNCERTAIN_LABEL
        self.__is_hierarchical_training: bool = Config.IS_HIERARCHICAL_TRAINING
        self.__hierarchical_labels: List[str] = Config.HIERARCHICAL_LABELS
        self.__train_val_data_dirs: List[str] = [Config.DATA_DIR]  # can add extra paths if needed
        self.__train_csv_path: str = Config.CSV_PATH
        self.__val_csv_path: str = Config.VAL_CSV_PATH
        self.__test_data_dir: str = Config.TEST_DATA_DIR
        self.__test_csv_path: str = Config.TEST_CSV_PATH
        self.__batch_size: int = Config.BATCH_SIZE
        self.__image_size: Tuple[int, int] = Config.IMAGE_SIZE
        self.__unique_labels: List[str] = Config.PATHOLOGY_LABELS
        self.__preprocess_unit = Config.BASE_MODEL_PREPROCESS_UNIT if Config.IS_BASE_MODEL_PREPROCESS else None
        self.__df_dict = self.__read_data_into_df()
        self.__unique_labels, self.__df_dict = self.__update_unique_labels_and_dfs(self.__df_dict)
        self.__unique_labels_index_dict = {label: idx for idx, label in enumerate(self.__unique_labels)}
        logger.info(f'Label indices: {self.__unique_labels_index_dict}')
        self.__num_train: int = len(self.__df_dict['train'])
        self.__num_class_positive: Dict[str, int] = self.__df_dict['train'][self.__unique_labels].sum(axis=0).to_dict()
        self.__class_weights: Dict[str, Dict[int, float]] = ModelUtils.calculate_class_weights(self.__num_train, self.__num_class_positive) if Config.USE_CLASS_WEIGHTS else None
        self.__set_hierarchy_indices()  # Set up hierarchy based on hierarchy_num
        self.__data_gen_dict = self.__data_generators()

    def __set_hierarchy_indices(self):
        """
        Set the parent and child indices.
        """
        self.__parent_indices = [
            self.__unique_labels_index_dict['Abnormal'],
            self.__unique_labels_index_dict['Opacity'],
            self.__unique_labels_index_dict['Fluid Accumulation'],
            self.__unique_labels_index_dict['Missing Lung Tissue'],
            self.__unique_labels_index_dict['Cardiac'],
            self.__unique_labels_index_dict['Other'],
            self.__unique_labels_index_dict['Enlarged Cardiomediastinum'],
            self.__unique_labels_index_dict['Consolidation']
        ]
        self.__child_indices = [
            [self.__unique_labels_index_dict['Opacity'],
             self.__unique_labels_index_dict['Cardiac'],
             self.__unique_labels_index_dict['Other']],  # Children of Abnormal

            [self.__unique_labels_index_dict['Lung Opacity'],
             self.__unique_labels_index_dict['Lung Lesion'],
             self.__unique_labels_index_dict['Fluid Accumulation'],
             self.__unique_labels_index_dict['Missing Lung Tissue']],  # Children of Opacity

            [self.__unique_labels_index_dict['Edema'],
             self.__unique_labels_index_dict['Consolidation'],
             self.__unique_labels_index_dict['Pleural Effusion'],
             self.__unique_labels_index_dict['Pneumonia']],  # Children of Fluid Accumulation

            [self.__unique_labels_index_dict['Atelectasis'],
             self.__unique_labels_index_dict['Pneumothorax'],
             self.__unique_labels_index_dict['Pleural Other']],  # Children of Missing Lung Tissue

            [self.__unique_labels_index_dict['Enlarged Cardiomediastinum'],
             self.__unique_labels_index_dict['Cardiomegaly']],  # Children of Cardiac

            [self.__unique_labels_index_dict['Fracture'],
             self.__unique_labels_index_dict['Support Devices']],  # Children of Other

            [self.__unique_labels_index_dict['Cardiomegaly']],  # Children of Enlarged Cardiomediastinum

            [self.__unique_labels_index_dict['Pneumonia']]  # Children of Consolidation

        ]

    def __file_exists_in_directories(self, relative_path: str) -> bool:
        """
        Check if a file exists in any of the specified directories.

        Parameters:
        relative_path (str): The relative path of the file.

        Returns:
        bool: True if the file exists in any of the directories, False otherwise.
        """
        return any(os.path.exists(os.path.join(directory, relative_path)) for directory in self.__train_val_data_dirs)

    def __add_uncertain_label_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add the indeterminate label to the dataframe if required.

        Parameters:
        df (pd.DataFrame): The dataframe to modify.

        Returns:
        pd.DataFrame: The modified dataframe with the indeterminate label added.
        """
        df[self.__uncertain_label] = (df[self.__unique_labels] == 0).all(axis=1).astype(float)
        return df

    @staticmethod
    def __add_hierarchical_labels_to_df(df_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add the parent label to the dataframe if required.

        Parameters:
        df (pd.DataFrame): The dataframe to modify.

        Returns:
        pd.DataFrame: The modified dataframe with the conditional labels added.
        """
        logger.info(f'Adding hierarchical labels to {df_name}_df')
        for index in tqdm(df.index):
            if not df.loc[index, ['No Finding', Config.UNCERTAIN_LABEL]].any():
                df.loc[index, 'Abnormal'] = 1.
            if df.loc[index, ['Enlarged Cardiomediastinum', 'Cardiomegaly']].any():
                df.loc[index, 'Cardiac'] = 1.
            if df.loc[index, ['Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Pleural Effusion', 'Atelectasis', 'Pneumothorax', 'Pleural Other']].any():
                df.loc[index, 'Opacity'] = 1.
            if df.loc[index, ['Edema', 'Consolidation', 'Pneumonia', 'Pleural Effusion']].any():
                df.loc[index, 'Fluid Accumulation'] = 1.
            if df.loc[index, ['Atelectasis', 'Pneumothorax', 'Pleural Other']].any():
                df.loc[index, 'Missing Lung Tissue'] = 1.
            if df.loc[index, ['Fracture', 'Support Devices']].any():
                df.loc[index, 'Other'] = 1.
        return df.fillna(0.)

    def __update_unique_labels_and_dfs(self, df_dict: Dict[str, pd.DataFrame]) -> Tuple[
        List[str], Dict[str, pd.DataFrame]]:
        """
        Add auxiliary (uncertain, and hierarchical | parent) labels to the dataframe.

        Parameters:
        df_dict (Dict[str, pd.DataFrame]): A dictionary of train, val, and test dataframes.

        Returns:
        updated_unique_labels, updated_df_dict (Tuple[List[str], Dict[str, pd.DataFrame]]): A list of unique labels and a dictionary of the corresponding labels.
        """
        updated_df_dict = df_dict
        updated_unique_labels = self.__unique_labels.copy()
        hierarchy_label_dir = f'data/hierarchical_labels/hierarchy_wu' if self.__is_uncertain_label else f'data/hierarchical_labels/hierarchy_wou'
        os.makedirs(hierarchy_label_dir, exist_ok=True)
        train_labels_file = os.path.join(hierarchy_label_dir, 'train_labels.csv')
        val_labels_file = os.path.join(hierarchy_label_dir, 'val_labels.csv')
        test_labels_file = os.path.join(hierarchy_label_dir, 'test_labels.csv')

        if (not os.path.isfile(train_labels_file) or
                not os.path.isfile(val_labels_file) or
                not os.path.isfile(test_labels_file)):
            if self.__is_uncertain_label:
                logger.info(f'self.__is_uncertain_label is set to {self.__is_uncertain_label}. '
                            f'"{self.__uncertain_label}" will be added to the labels.')
                updated_df_dict = {key: self.__add_uncertain_label_to_df(df) for (key, df) in df_dict.items()}
                updated_unique_labels.append(self.__uncertain_label)

            if self.__is_hierarchical_training:
                logger.info(f'self.__is_hierarchical_training is set to {self.__is_hierarchical_training}. '
                            f'"{self.__hierarchical_labels}" will be added to the labels.')
                updated_df_dict = {key: self.__add_hierarchical_labels_to_df(key, df) for (key, df) in
                                   updated_df_dict.items()}
                updated_unique_labels.extend(self.__hierarchical_labels)

            for (key, df) in updated_df_dict.items():
                df.to_csv(os.path.join(hierarchy_label_dir, f'{key}_labels.csv'), index=False)
        else:
            updated_df_dict['train'] = pd.read_csv(train_labels_file)
            updated_df_dict['val'] = pd.read_csv(val_labels_file)
            updated_df_dict['test'] = pd.read_csv(test_labels_file)
            try:
                if ((self.__is_uncertain_label and
                     self.__uncertain_label in updated_df_dict['train'].columns) and
                        self.__uncertain_label not in updated_unique_labels):
                    updated_unique_labels.append(self.__uncertain_label)
                if (self.__is_hierarchical_training and
                        set(self.__hierarchical_labels).issubset(updated_df_dict['train'].columns) and
                        not set(self.__hierarchical_labels).issubset(updated_unique_labels)):
                    updated_unique_labels.extend(self.__hierarchical_labels)
            except KeyError as e:
                logger.error(f'Error updating the dataframes and/or unique labels: {e}')
        logger.info(f'Classes updated: {updated_unique_labels}')
        return updated_unique_labels, updated_df_dict

    def __read_data_into_df(self) -> Dict[str, pd.DataFrame]:
        """
        Read and preprocess the data from CSV files into dataframes.

        Returns:
        Dict[str, pd.DataFrame]: Dictionary containing train, validation, and test dataframes.
        """
        logger.info('Reading data from CSV files')
        try:
            df_dict = {'train': pd.read_csv(self.__train_csv_path),
                       'val': pd.read_csv(self.__val_csv_path),
                       'test': pd.read_csv(self.__test_csv_path)}
            df_dict['train'] = df_dict['train'][df_dict['train']['Path'].apply(self.__file_exists_in_directories)]
            logger.info(f'Total number of train data: {len(df_dict["train"])}')
            df_dict['val'] = df_dict['val'][df_dict['val']['Path'].apply(lambda x: os.path.exists(os.path.join(self.__train_val_data_dirs[0], x)))]
            logger.info(f'Total number of validation data: {len(df_dict["val"])}')
            df_dict['test'] = df_dict['test'][df_dict['test']['Path'].apply(lambda x: os.path.exists(os.path.join(self.__test_data_dir, x)))]
            logger.info(f'Total number of test data: {len(df_dict["test"])}')

        except Exception as e:
            logger.error(f'Error reading data: {e}')
            raise
        return df_dict

    def __compute_penalties(self, df: pd.DataFrame, parent_indices: List[int], child_indices: List[List[int]],
                            epsilon: float = 1e-6) -> Dict[int, Dict[int, float]]:
        """
        Compute penalties based on empirical conditional probabilities with Laplace smoothing.

        Parameters:
        df (pd.DataFrame): DataFrame containing the training data.
        parent_indices (List[int]): List of parent label indices.
        child_indices (List[List[int]]): List of lists of child label indices.
        epsilon (float): Small value for Laplace smoothing (default: 1e-6).

        Returns:
        Dict[int, Dict[int, float]]: Nested dictionary containing penalties for each parent-child pair.
        """
        penalties = {}
        for parent_idx, children in zip(parent_indices, child_indices):
            parent_label = self.__unique_labels[parent_idx]
            penalties[parent_idx] = {}
            parent_negative = df[df[parent_label] == 0]
            count_parent_negative = len(parent_negative)
            for child_idx in children:
                child_label = self.__unique_labels[child_idx]
                inconsistent_cases = parent_negative[parent_negative[child_label] == 1]
                count_inconsistent = len(inconsistent_cases) + epsilon
                count_parent_negative_smooth = count_parent_negative + epsilon * 2  # Apply Laplace smoothing
                probability = count_inconsistent / count_parent_negative_smooth
                penalty = 1.0 - probability  # Higher penalty if fewer inconsistencies
                penalties[parent_idx][child_idx] = penalty
        logger.info(f'Computed penalties: {penalties}')
        return penalties

    def __data_generators(self) -> Dict[str, tf.data.Dataset]:
        """
        Get the train, validation, and test data generators.

        Returns:
        Dict[str, tf.data.Dataset]: Dictionary containing train, validation, and test data generators.
        """
        train_gen = CheXpertDataset(df=self.__df_dict['train'],
                                    unique_labels=self.__unique_labels,
                                    input_folders=self.__train_val_data_dirs,
                                    batch_size=self.__batch_size,
                                    shuffle=True,
                                    preprocess=True,
                                    augment=True,
                                    preprocess_unit=self.__preprocess_unit).get_dataset()

        val_gen = CheXpertDataset(df=self.__df_dict['val'],
                                  unique_labels=self.__unique_labels,
                                  input_folders=[self.__train_val_data_dirs[0]],
                                  batch_size=self.__batch_size,
                                  shuffle=True,
                                  preprocess=True,
                                  augment=True,
                                  preprocess_unit=self.__preprocess_unit).get_dataset()

        test_gen = CheXpertDataset(df=self.__df_dict['test'],
                                   unique_labels=self.__unique_labels,
                                   input_folders=[self.__test_data_dir],
                                   batch_size=self.__batch_size,
                                   shuffle=False,
                                   preprocess=True,
                                   augment=False,
                                   preprocess_unit=self.__preprocess_unit).get_dataset()

        return {'train': train_gen, 'val': val_gen, 'test': test_gen}

    def get_data_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the train, validation, and test dataframes.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test dataframes.
        """
        return self.__df_dict['train'], self.__df_dict['val'], self.__df_dict['test']

    def get_unique_labels(self) -> List[str]:
        """
        Get the unique labels.

        Returns:
        List[str]: The unique labels.
        """
        return self.__unique_labels

    def get_unique_labels_index_dict(self) -> Dict[str, int]:
        """
        Get the unique labels dictionary.

        Returns:
        Dict[str, int]: The unique labels.
        """
        return self.__unique_labels_index_dict

    def get_num_classes(self) -> int:
        """
        Get the number of classes.

        Returns:
        int: The number of classes.
        """
        return len(self.__unique_labels)

    def get_data_generators(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Get the train, validation, and test data generators.

        Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Train, validation, and test data generators.
        """
        return self.__data_gen_dict['train'], self.__data_gen_dict['val'], self.__data_gen_dict['test']

    def get_class_weights(self) -> Dict[str, Dict[int, float]]:
        """
        Get the class weights for the training data.

        Returns:
        Dict[str, Dict[int, float]]: Class weights for each label.
        """
        return self.__class_weights

    def get_parent_child_indices(self) -> Tuple[List[int], List[List[int]]]:
        """
        Get the parent and child indices for hierarchical training.

        Returns:
        Tuple[List[int], List[List[int]]]: Parent indices and child indices.
        """
        return self.__parent_indices, self.__child_indices

    def get_penalties(self) -> Dict[int, Dict[int, float]]:
        """
        Get the computed penalties for the hierarchical loss function.

        Returns:
        Dict[int, Dict[int, float]]: Nested dictionary containing penalties for each parent-child pair.
        """
        return self.__compute_penalties(self.__df_dict['train'], self.__parent_indices, self.__child_indices)
