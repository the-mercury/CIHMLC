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
    seed = 2408
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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


def evaluate(data_loader: Optional[CheXpertDataLoader] = None,
             eval_model: Optional[KerasModel] = None,
             model_path: Optional[str] = None):
    """
    Evaluate the X-ray model.

    Parameters:
    data_loader (Optional[CheXpertDataLoader]): The CheXpert data loader.
    ev_model (Optional[keras.Model]): The model to evaluate. If None, a model will be loaded from model_path.
    model_path (Optional[str]): The path to the model to load. Ignored if ev_model is provided.
    """
    if model_path:
        ev_model = ModelUtils.load_model(model_path)
    elif eval_model:
        ev_model = eval_model
    else:
        logger.error('No model provided for evaluation.')
        return

    if not data_loader:
        data_loader = CheXpertDataLoader()
    else:
        data_loader = data_loader

    _, _, test_df = data_loader.get_data_frames()
    _, _, test_gen = data_loader.get_data_generators()
    unique_labels = data_loader.get_unique_labels()

    y_score = ev_model.predict(test_gen)

    try:
        roc_auc = roc_auc_score(test_df[unique_labels].values, y_score, average='weighted')
    except ValueError as e:
        roc_auc = 0.0
        logger.warning(f'AUROC calculation failed: {e}')

    logger.info(f'AUROC for {eval_model.name}: {roc_auc:.4f}')
    ModelUtils.get_roc_curve(unique_labels, y_score, test_df[unique_labels].values, roc_auc, ev_model.name)


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
