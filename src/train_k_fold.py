import os
import random
import time
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model as KerasModel
from sklearn.metrics import f1_score, roc_auc_score, multilabel_confusion_matrix, precision_score, recall_score
from sklearn.model_selection import KFold

from src.config import Config
from src.data.chexpert_data_loader import CheXpertDataLoader
from src.helpers.logger_config import LoggerConfig, LoggingCallback
from src.helpers.model_utils import ModelUtils
from src.x_ray_trainer import XRayTrainer

seed = 2408
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def get_preds(data_gen, model: KerasModel):
    y_pred = model.predict(data_gen)
    y_true = np.concatenate([y for _, y in data_gen], axis=0)
    return y_true, y_pred


def safe_macro_auc(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        try:
            if np.unique(y_true[:, i]).size == 2:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(auc)
        except:
            continue
    return np.mean(scores) if scores else 0.0


def save_kfold_results_csv(k_results: dict, save_path: str):
    rows = []
    for lam, result in k_results.items():
        for fold_data in result['folds']:
            row = {
                'lambda': lam,
                'fold': fold_data['fold'],
                'fold_f1_macro': fold_data['f1_macro'],
                'fold_f1_weighted': fold_data['f1_weighted'],
                'fold_auc_macro': fold_data['auc_macro'],
                'fold_auc_weighted': fold_data['auc_weighted'],
                'fold_time_sec': fold_data['time_sec'],
                'avg_f1_macro': result['macro_f1_avg'],
                'avg_f1_weighted': result['weighted_f1_avg'],
                'avg_auc_macro': result['macro_auc_avg'],
                'avg_auc_weighted': result['weighted_auc_avg']
            }
            for label, metrics in fold_data['label_metrics'].items():
                row[f'{label}_precision'] = metrics.get('precision', None)
                row[f'{label}_recall'] = metrics.get('recall', None)
                row[f'{label}_auc'] = metrics.get('auc', None)
                row[f'{label}_f1'] = metrics.get('f1', None)
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f'Saved k-fold results to CSV: {save_path}')


def train(train_gen, val_gen, xray_trainer: XRayTrainer, model_save_id=None) -> KerasModel:
    model = xray_trainer.get_model()
    call_backs = ModelUtils.get_callbacks(model.name)
    call_backs.append(LoggingCallback(logger=logger.info))

    model.fit(train_gen,
              epochs=Config.EPOCHS,
              validation_data=val_gen,
              batch_size=Config.BATCH_SIZE,
              callbacks=call_backs,
              class_weight=None,
              verbose=1)

    xray_trainer.save_model(save_id=model_save_id)
    return model


def k_fold_train(k_lambda_grid: list, k_seed: int = 2408, n_splits: int = 5) -> Tuple[float, dict]:
    output_dir = os.path.join(Config.TRAIN_LOG_DIR, f'kfold_lambda_tuning_{Config.START_TIME}')
    os.makedirs(output_dir, exist_ok=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=k_seed)
    k_best_lambda = None
    best_avg_f1 = -1.0
    k_results = {}

    data_loader = CheXpertDataLoader()
    train_df, val_df, _ = data_loader.get_data_frames()
    train_val_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    for lam in k_lambda_grid:
        l_output_dir = os.path.join(output_dir, f'lambda_{lam}')
        os.makedirs(l_output_dir, exist_ok=True)
        logger.info(f'>>> TUNING λ = {lam} <<<')
        macro_f1_scores = []
        weighted_f1_scores = []
        weighted_auc_scores = []
        macro_auc_scores = []
        lambda_result = {'folds': []}

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df), start=1):
            k_output_dir = os.path.join(l_output_dir, f'lambda{lam}_fold{fold}')
            os.makedirs(k_output_dir, exist_ok=True)
            logger.info(f'>>> λ = {lam}, Fold {fold}/{n_splits} <<<')
            k_train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
            penalties = data_loader.get_penalties(k_train_df)
            k_val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
            logger.info(f'# train: {len(k_train_df)} | # val: {len(k_val_df)}')

            xray_trainer = XRayTrainer(data_loader=data_loader,
                                       train_df=train_val_df,
                                       penalties=penalties,
                                       penalty_scale=lam,
                                       use_computed_penalties=True)

            train_gen = data_loader.get_data_generator_for_df(k_train_df)
            val_gen = data_loader.get_data_generator_for_df(k_val_df)

            logger.info(f'λ = {lam}, Fold {fold} training started...')
            model_save_id = f'{xray_trainer.get_model_name()}_fold_{fold}'
            start_time = time.time()
            model = train(train_gen, val_gen, xray_trainer, model_save_id=model_save_id)
            elapsed = time.time() - start_time
            logger.info(f'λ = {lam}, Fold {fold} training finished!')

            y_true, y_pred = get_preds(val_gen, model)
            f1_macro = f1_score(y_true, y_pred > 0.5, average='macro')
            f1_weighted = f1_score(y_true, y_pred > 0.5, average='weighted')
            auc_macro = safe_macro_auc(y_true, y_pred)
            try:
                auc_weighted = roc_auc_score(y_true, y_pred, average='weighted')
            except:
                auc_weighted = np.nan
            conf_matrix = multilabel_confusion_matrix(y_true, y_pred > 0.5)

            label_metrics = {}
            labels = data_loader.get_unique_labels()
            for i, label in enumerate(labels):
                try:
                    if np.unique(y_true[:, i]).size == 2:
                        precision = precision_score(y_true[:, i], y_pred[:, i] > 0.5, zero_division=np.nan)
                        recall = recall_score(y_true[:, i], y_pred[:, i] > 0.5, zero_division=np.nan)
                        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                        f1 = f1_score(y_true[:, i], y_pred[:, i] > 0.5, zero_division=np.nan)
                        label_metrics[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}
                except:
                    continue
            fold_log = {
                'fold': fold,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'auc_macro': auc_macro,
                'auc_weighted': auc_weighted,
                'time_sec': elapsed,
                'label_metrics': label_metrics
            }

            pd.DataFrame(y_true, columns=labels).to_csv(
                os.path.join(k_output_dir, f'lambda_{lam}_fold_{fold}_y_true.csv'), index=False
            )
            pd.DataFrame(y_pred, columns=labels).to_csv(
                os.path.join(k_output_dir, f'lambda_{lam}_fold_{fold}_y_pred.csv'), index=False
            )

            # Flatten and label confusion matrix entries
            for i, label in enumerate(labels):
                cm = conf_matrix[i]
                cm_df = pd.DataFrame(cm, columns=[f'{label}_pred_0', f'{label}_pred_1'],
                                     index=[f'{label}_true_0', f'{label}_true_1'])
                cm_df.to_csv(os.path.join(k_output_dir, f'lambda_{lam}_fold_{fold}_{label}_conf_matrix.csv'))

            logger.info(f'λ = {lam}, Fold {fold} | F1: {f1_macro:.4f}, AUROC: {auc_macro:.4f}')
            macro_f1_scores.append(f1_macro)
            weighted_f1_scores.append(f1_weighted)
            macro_auc_scores.append(auc_macro)
            weighted_auc_scores.append(auc_weighted)
            lambda_result['folds'].append(fold_log)
            pd.DataFrame.from_dict(fold_log).to_csv(os.path.join(k_output_dir, f'lambda_{lam}_fold_{fold}_results.csv'))

        avg_f1_macro = np.mean(macro_f1_scores)
        avg_f1_weighted = np.mean(weighted_f1_scores)
        avg_auc_macro = np.mean(macro_auc_scores)
        avg_auc_weighted = np.mean(weighted_auc_scores)
        logger.info(
            f'λ = {lam} → Avg F1 Macro: {avg_f1_macro:.4f}, Avg F1 Weighted: {avg_f1_weighted:.4f}, '
            f'Avg AUROC Macro: {avg_auc_macro:.4f}, Avg AUROC Weighted: {avg_auc_weighted:.4f}'
        )
        lambda_result.update({'macro_f1_avg': avg_f1_macro, 'macro_auc_avg': avg_auc_macro})
        lambda_result.update({'weighted_f1_avg': avg_f1_weighted, 'weighted_auc_avg': avg_auc_weighted})

        k_results[lam] = lambda_result
        pd.DataFrame.from_dict(lambda_result).to_csv(os.path.join(l_output_dir, f'lambda_{lam}_results.csv'))

        if avg_f1_macro > best_avg_f1:
            best_avg_f1 = avg_f1_macro
            k_best_lambda = lam
    save_kfold_results_csv(k_results, os.path.join(output_dir, 'lambda_kfold_results.csv'))
    logger.info(f'>>> BEST λ: {k_best_lambda} with Avg F1: {best_avg_f1:.4f} <<<')
    return k_best_lambda, k_results


def final_train_and_eval(best_lambda: float):

    data_loader = CheXpertDataLoader()
    # train_df, val_df, test_df = data_loader.get_data_frames()
    train_gen, val_gen, test_gen = data_loader.get_data_generators()
    # test_gen = data_loader.get_data_generator_for_df(test_df, shuffle=False, augment=False)

    xray_trainer = XRayTrainer(hierarchy_penalty=None,
                               penalty_scale=best_lambda,
                               use_computed_penalties=True)

    model = train(train_gen, val_gen, xray_trainer, model_save_id='final')
    y_true, y_pred = get_preds(test_gen, model)

    try:
        test_auc_weighted = roc_auc_score(y_true, y_pred, average='weighted')
    except ValueError as e:
        test_auc_weighted = 0.0
        logger.warning(f'AUROC calculation failed: {e}')

    test_f1_weighted = f1_score(y_true, y_pred > 0.5, average='weighted')
    test_f1_macro = f1_score(y_true, y_pred > 0.5, average='macro')
    test_auc_macro = safe_macro_auc(y_true, y_pred)

    logger.info(f'Final Test Weighted F1 with λ = {best_lambda}: {test_f1_weighted:.4f}')
    logger.info(f'Final Test Macro F1 with λ = {best_lambda}: {test_f1_macro:.4f}')
    logger.info(f'Final Test Weighted AUROC with λ = {best_lambda}: {test_auc_weighted:.4f}')
    logger.info(f'Final Test Macro AUROC with λ = {best_lambda}: {test_auc_macro:.4f}')

    ModelUtils.get_roc_curve(data_loader.get_unique_labels(), y_pred, y_true, test_auc_weighted, model.name)


if __name__ == '__main__':
    Config.START_TIME = time.strftime('%b_%d_%Y_%H%M')
    logger = LoggerConfig().get_logger('TRAIN_KFOLD', Config.TRAIN_LOG_DIR)

    lambda_grid = [0.0, 0.3, 0.5, 0.7, 1.0]
    final_lambda, results = k_fold_train(lambda_grid, seed)
    final_train_and_eval(final_lambda)
