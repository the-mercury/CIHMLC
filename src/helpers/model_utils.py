import logging
import os
from typing import List, Optional, Dict

import numpy as np
import tensorflow as tf
from keras import Model as KerasModel, utils as keras_utils, callbacks as keras_callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.models import load_model
from matplotlib import colormaps, pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import Config
from src.hierarchical_binary_cross_entropy import HierarchicalBinaryCrossEntropy

# Logging setup
logger = logging.getLogger('ModelUtils')


class ModelUtils:
    """
    Utility class for model operations such as loading models, handling class weights,
    generating Grad-CAM heatmaps, and plotting ROC curves.
    """

    @staticmethod
    def load_model(model_path: str) -> KerasModel:
        """
        Load a Keras model from the specified file path.

        Parameters:
        model_path (str): The path to the model file.

        Returns:
        Model: The loaded Keras model.
        """
        model = load_model(model_path, custom_objects={'HierarchicalBinaryCrossentropy': HierarchicalBinaryCrossEntropy})
        model.trainable = False
        return model

    @staticmethod
    def load_weights(model: KerasModel, weights_path: str) -> None:
        """
        Load weights into a Keras model from the specified file path.

        Parameters:
        model (Model): The Keras model to load weights into.
        weights_path (str): The path to the weights file.
        """
        model.load_weights(weights_path)

    @staticmethod
    def overlay_heatmap(heatmap: tf.Tensor, img: np.ndarray, clip_threshold: float = None, alpha: float = 0.7, cmap: str = 'hot_r') -> np.ndarray:
        """
        Overlay a heatmap onto an image.

        Parameters:
        heatmap (tf.Tensor): The heatmap to overlay.
        img (np.ndarray): The original image.
        alpha (float): The transparency level for the heatmap. Defaults to 0.4.
        cmap (str): The colormap to use for the heatmap. Defaults to "jet".

        Returns:
        np.ndarray: The image with the heatmap overlay.
        """
        heatmap = np.uint8(255 * heatmap)
        color_map = colormaps[cmap](np.arange(256))[:, :3]
        color_mapped_heatmap = color_map[heatmap]
        color_mapped_heatmap = keras_utils.array_to_img(color_mapped_heatmap)
        color_mapped_heatmap = color_mapped_heatmap.resize((img.shape[1], img.shape[0]))
        color_mapped_heatmap = keras_utils.img_to_array(color_mapped_heatmap)
        color_mapped_heatmap = np.where(color_mapped_heatmap > int(clip_threshold * 255), color_mapped_heatmap, 0) if clip_threshold else color_mapped_heatmap

        superimposed_img = (alpha * color_mapped_heatmap) + (255 * img)
        # superimposed_img = np.clip(superimposed_img, 0, 255)
        return np.array(keras_utils.array_to_img(superimposed_img))

    @staticmethod
    def __get_stage_item(stage: int, stage_map: dict, item_name: str) -> any:
        if stage not in stage_map:
            raise ValueError(f'Invalid stage: {stage}')
        logger.info(f'Original {item_name}: {stage_map[stage]}')
        return stage_map[stage]

    @staticmethod
    def __get_model_item(base_model: Optional[str], items: dict, default_items: dict, item_type: str) -> any:
        if base_model:
            if base_model not in items:
                raise ValueError(f'Invalid {item_type}: {base_model}')
            return items[base_model]

        return default_items

    @staticmethod
    def get_base_model(base_model: Optional[str]) -> KerasModel:
        """
        Get the base model based on the stage or user input.

        Parameters:
        stage (int): The stage of the multi-stage classifier.
        base_model (Optional[str]): The base model specified by the user.

        Returns:
        Model: The selected base model.
        """
        base_models = Config.BASE_MODEL
        default_base_models = {
            1: base_models['efficientnetb0'],
            2: base_models['densenet121'],
            21: base_models['efficientnetb0'],
            22: base_models['efficientnetb0'],
            23: base_models['efficientnetb0'],
            24: base_models['efficientnetb0'],
            3: base_models['densenet121'],
        }
        return ModelUtils.__get_model_item(base_model, base_models, default_base_models, 'base model')

    @staticmethod
    def get_preprocess_unit(base_model: Optional[str]) -> callable:
        """
        Get the preprocessing function based on the stage.

        Parameters:
        stage (int): The stage of the multi-stage classifier.
        base_model (Optional[str]): The base model specified by the user.

        Returns:
        callable: The preprocessing function for the base model.
        """
        preprocess_units = Config.BASE_MODEL_PREPROCESS_UNIT
        default_preprocess_units = {
            1: preprocess_units['efficientnetb0'],
            2: preprocess_units['densenet121'],
            21: preprocess_units['efficientnetb0'],
            22: preprocess_units['efficientnetb0'],
            23: preprocess_units['efficientnetb0'],
            24: preprocess_units['efficientnetb0'],
            3: preprocess_units['densenet121'],
        }
        return ModelUtils.__get_model_item(base_model, preprocess_units, default_preprocess_units, 'preprocess unit')

    @staticmethod
    def get_roc_curve(labels: List[str], predicted_vals: np.ndarray, real_vals: np.ndarray, roc_auc: float, model_name: str) -> List[float]:
        """
        Generate and save ROC curve for each label and return the AUC values.

        Parameters:
        labels (List[str]): List of label names.
        predicted_vals (np.ndarray): Predicted values from the model.
        real_vals (np.ndarray): Ground truth values.
        roc_auc (float): The overall ROC AUC score for the model.
        model_name (str): The name of the model.

        Returns:
        List[float]: A list of ROC AUC values for each label.
        """
        auc_roc_vals = []
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(len(labels)):
            try:
                gt = real_vals[:, i]
                pred = predicted_vals[:, i]
                auc_roc = roc_auc_score(gt, pred, average='weighted')
                auc_roc_vals.append(auc_roc)
                fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            except ValueError:
                logger.error(
                    f'Error in generating ROC curve for {labels[i]}. Dataset lacks enough examples for the label!')
                continue  # Skip plotting for this label if calculation fails

            ax.plot([0, 1], [0, 1], 'k--')
            ax.plot(fpr_rf, tpr_rf, label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        plt.suptitle(f'{model_name}')
        plt.title(f'AUROC curve--weighted mean: {roc_auc:.4f}')
        plt.legend(loc='best')
        fig_id = f'{model_name}--AUROC_{roc_auc:.4f}.png'
        plot_save_dir = os.path.join(f'{Config.LOG_DIR}/plots/{model_name}')
        os.makedirs(plot_save_dir, exist_ok=True)
        fig_save_path = os.path.join(f'{plot_save_dir}', fig_id)
        fig.savefig(fig_save_path)
        plt.close(fig)
        logger.info(f'Saved AUROC curve as {fig_save_path}')
        return auc_roc_vals

    @staticmethod
    def calculate_class_weights(total_counts: int, class_positive_counts: Dict[str, int]) -> Dict[str, Dict[int, float]]:
        """
        Calculate the class weights based on the positive counts of each class.

        Parameters:
        total_counts (int): Total number of samples.
        class_positive_counts (Dict[str, int]): Dictionary with positive counts for each class.

        Returns:
        Dict[str, Dict[int, float]]: Calculated class weights.
        """
        class_names = list(class_positive_counts.keys())
        label_counts = np.array(list(class_positive_counts.values()))
        class_weights_list = []

        def _get_single_class_weight(counts: int, positive_counts: int, multiply: int = 1) -> Dict[int, float]:
            denominator = (counts - positive_counts) * multiply + positive_counts
            return {
                0: positive_counts / denominator,
                1: (denominator - positive_counts) / denominator,
            }

        for i, class_name in enumerate(class_names):
            class_weights_list.append(_get_single_class_weight(total_counts, label_counts[i]))
        class_weights = {cn: cw for cn, cw in zip(class_names, class_weights_list)}
        logger.info(f'Class weights: {class_weights}')
        return class_weights

    @staticmethod
    def get_callbacks(model_name: str) -> List[keras_callbacks.Callback]:
        """
        Define and return the list of callbacks for model training.

        Parameters:
        model_name (str): The model name used in saving the checkpoints.

        Returns:
        List[keras.callbacks.Callback]: List of Keras callbacks.
        """
        best_loss_model_id = f'{model_name}--best_loss_model_ckpt.keras'
        best_auc_model_id = f'{model_name}--best_auc_model_ckpt.keras'

        def lr_scheduler(epoch: int, lr: float) -> float:
            if epoch < Config.EPOCHS:
                return lr
            else:
                return lr * np.exp(-0.1)

        best_loss_model_ckpt = ModelCheckpoint(
            os.path.join(Config.NEW_CHKPT_PATH, 'best_loss', best_loss_model_id),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        best_auc_model_ckpt = ModelCheckpoint(
            os.path.join(Config.NEW_CHKPT_PATH, 'best_auc', best_auc_model_id),
            monitor='val_AUC',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.9,
            patience=1,
            min_lr=1e-8,
            mode='min',
            verbose=1
        )

        schedule_lr = LearningRateScheduler(
            lr_scheduler,
            verbose=1
        )

        tensorboard_callback = keras_callbacks.TensorBoard(
            log_dir=f"{Config.LOG_DIR}/tensorboard/{model_name}", histogram_freq=1
        )

        callbacks_list = [
            best_loss_model_ckpt,
            best_auc_model_ckpt,
            early_stop,
            reduce_lr,
            # schedule_lr,
            tensorboard_callback,
        ]
        return callbacks_list
