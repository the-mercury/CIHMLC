import random
import time
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from keras import Model as KerasModel, Model
from sklearn.metrics import roc_auc_score

from src.config import Config
from src.data.chexpert_data_loader import CheXpertDataLoader
from src.helpers.logger_config import LoggerConfig, LoggingCallback
from src.helpers.model_utils import ModelUtils
from src.x_ray_trainer import XRayTrainer

seed = 2408
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def train(tr_hierarchy_penalty: Optional[float] = None,
          tr_penalty_scale: Optional[float] = None,
          tr_use_computed_penalties: bool = False) -> Union[Tuple[Model, CheXpertDataLoader], tuple[None, None]]:
    """
    Train the X-ray model.

    Parameters:
    tr_hierarchy_penalty (Optional[float]): The penalty parameter for hierarchy prediction.
    tr_penalty_scale: (Optional[float]): The penalty parameter for penalty scaling.
    tr_use_computed_penalties: bool: Whether to use computed penalties.

    Returns:
    Optional[keras.Model]: The trained Keras model, or None if an error occurs.
    """

    try:
        xray_trainer = XRayTrainer(hierarchy_penalty=tr_hierarchy_penalty,
                                   penalty_scale=tr_penalty_scale,
                                   use_computed_penalties=tr_use_computed_penalties)
        data_loader = xray_trainer.get_data_loader()
        train_gen, val_gen, test_gen = data_loader.get_data_generators()
        # DataUtils.visualize_tf_dataset(train_gen)

        model = xray_trainer.get_model()
        model_config = xray_trainer.get_config()

        call_backs = ModelUtils.get_callbacks(model.name)
        call_backs.append(LoggingCallback(logger=logger.info))

        logger.info(f'\n\n>>> Compiled Config: <<<\n{model_config.to_string()}\n')

        logger.info('Training started!')
        model.fit(train_gen,
                  epochs=Config.EPOCHS,
                  validation_data=val_gen,
                  batch_size=Config.BATCH_SIZE,
                  callbacks=call_backs,
                  class_weight=None,
                  verbose=1)
        logger.info('Training finished!')
        xray_trainer.save_model()
        return model, data_loader

    except Exception as e:
        logger.exception(f'Exception during training: {e}')
        return None, None


def _freeze_batchnorm_inference(model):
    """Make BatchNorm run in inference mode even when we pass training=True."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def _iter_once(gen):
    """Yield batches from either a Keras Sequence, tf.data.Dataset, or plain generator."""
    # Keras Sequence (has __len__ and __getitem__)
    if hasattr(gen, '__len__') and hasattr(gen, '__getitem__'):
        for i in range(len(gen)):
            batch = gen[i]
            yield batch if isinstance(batch, (tuple, list)) else (batch, None)
    else:
        # tf.data or Python generator
        for batch in gen:
            yield batch if isinstance(batch, (tuple, list)) else (batch, None)


def predict_mcd(model, test_gen, passes=30, verbose=0):
    """
    Returns:
      mean_probs: [N, L]  — mean probability over passes
      std_probs:  [N, L]  — std-dev over passes
      all_probs:  [T, N, L] (optional; returned for convenience)
    Notes:
      - Ensure test_gen has NO shuffle/repeat.
      - BatchNorm is frozen to inference; Dropout is active (training=True).
    """
    _freeze_batchnorm_inference(model)

    # Collect predictions for each pass
    preds_list = []
    for t in range(passes):
        if verbose:
            print(f"[MCD] pass {t + 1}/{passes}", end="\r")
        batch_preds = []
        for x, _ in _iter_once(test_gen):
            # Activate dropout
            y = model(x, training=True)
            y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
            # if from_logits:
            #     y = 1.0 / (1.0 + np.exp(-y))  # sigmoid
            batch_preds.append(y)
        preds_t = np.concatenate(batch_preds, axis=0)  # [N, L]
        preds_list.append(preds_t)

    all_probs = np.stack(preds_list, axis=0)  # [T, N, L]
    mean_probs = all_probs.mean(axis=0)  # [N, L]
    std_probs = all_probs.std(axis=0)  # [N, L]
    return mean_probs, std_probs, all_probs


def evaluate(data_loader: Optional[CheXpertDataLoader] = None,
             eval_model: Optional[KerasModel] = None,
             model_path: Optional[str] = None,
             mode: str = 'test', is_mcd: bool = False) -> float:
    """
    Evaluate the X-ray model.

    Parameters:
    data_loader (Optional[CheXpertDataLoader]): The CheXpert data loader.
    ev_model (Optional[keras.Model]): The model to evaluate. If None, a model will be loaded from model_path.
    model_path (Optional[str]): The path to the model to load. Ignored if ev_model is provided.
    mode (str): Whether to evaluate on evaluation or test set.
    is_mcd (bool): Whether the evaluation uses MCD or not.
    """
    if model_path:
        eval_model = ModelUtils.load_model(model_path)
    elif not eval_model:
        logger.error('No model provided for evaluation.')
        return 0.0

    if not data_loader:
        data_loader = CheXpertDataLoader()

    if mode == 'val':
        _, val_df, _ = data_loader.get_data_frames()
        _, val_gen, _ = data_loader.get_data_generators()
        y_true = val_df[data_loader.get_unique_labels()].values
        if is_mcd:
            y_score, y_score_std, _ = predict_mcd(
                eval_model,
                val_gen,
                passes=50,  # tweak as you like (30–100 common)
                # from_logits=False,  # set True if your model outputs logits
                verbose=1
            )
            np.save(y_score, 'default_model_files/scores/test_val_scores/y_pred_val_mean.npy')
            np.save(y_score_std, 'default_model_files/scores/test_val_scores/y_pred_val_std.npy')
        else:
            y_score = eval_model.predict(val_gen)
            np.save(y_score, 'default_model_files/scores/test_val_scores/y_pred_val.npy')
        np.save(y_true, 'default_model_files/scores/y_true_val.npy')
    else:
        _, _, test_df = data_loader.get_data_frames()
        _, _, test_gen = data_loader.get_data_generators()
        y_true = test_df[data_loader.get_unique_labels()].values
        if is_mcd:
            y_score, y_score_std, _ = predict_mcd(
                eval_model,
                test_gen,
                passes=50,  # tweak as you like (30–100 common)
                # from_logits=False,  # set True if your model outputs logits
                verbose=1
            )
            np.save(y_score, 'default_model_files/scores/test_val_scores/y_pred_test_mean.npy')
            np.save(y_score_std, 'default_model_files/scores/test_val_scores/y_pred_test_std.npy')
        else:
            y_score = eval_model.predict(test_gen)
            np.save(y_score, 'default_model_files/scores/test_val_scores/y_pred_test.npy')
        np.save(y_true, 'default_model_files/scores/y_true_test.npy')

    try:
        roc_auc = roc_auc_score(y_true, y_score, average='weighted')
    except ValueError as e:
        roc_auc = 0.0
        logger.warning(f'AUROC calculation failed: {e}')

    logger.info(f'AUROC ({mode}) for {eval_model.name}: {roc_auc:.4f}')
    if mode == 'test':
        ModelUtils.get_roc_curve(data_loader.get_unique_labels(), y_score, y_true, roc_auc, eval_model.name)

    return roc_auc


if __name__ == '__main__':

    penalty_scale, penalty = (1.0, None) if Config.USE_COMPUTED_PENALTIES else (None, 0.5)
    penalty_type = f'fixed_{penalty}' if penalty is not None else 'DataDriven'

    Config.START_TIME = time.strftime('%a_%d_%b_%Y_%H%M')  # overriding the config start time
    logger = LoggerConfig().get_logger('TRAIN', Config.TRAIN_LOG_DIR)
    logger.info(f'\n\n>>> {str.upper("training")} -- {str.upper("hierarchical")} -- {str.upper("penalty: ")}{penalty_type} -- {str.upper("scale factor: ")}{penalty_scale} <<<\n')

    start_time = time.time()
    new_model, new_data_loader = train(tr_hierarchy_penalty=penalty,
                                       tr_penalty_scale=penalty_scale,
                                       tr_use_computed_penalties=Config.USE_COMPUTED_PENALTIES)
    elapsed_time = time.time() - start_time
    logger.info(f'Training completed in {elapsed_time:.2f}s')

    logger.info(f'\n\n>>> {str.upper("evaluating")} -- {str.upper("hierarchical")} -- {str.upper("penalty: ")}{penalty_type} -- {str.upper("scale factor: ")}{penalty_scale} <<<\n')
    evaluate(data_loader=new_data_loader, eval_model=new_model)
