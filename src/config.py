import os
import time
from typing import Dict, List, Tuple, Any

from keras import Input
from keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess_input
# from keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess_input
# from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess_input
# from keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess_input
# from keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess_input


class Config:
    """
    Configuration and setup class.
    """
    START_TIME: str = time.strftime('%a_%d_%b_%Y_%H%M')

    # Device configuration
    DEVICE: str = os.getenv('DEVICE', 'CPU')

    # App-specific settings
    ENVIRONMENT = 'development'
    ALLOWED_ORIGINS = (
        ["*"] if ENVIRONMENT == 'development' else ['https://your-frontend.com']
    )
    APP: str = 'src.cxr_inference_app:app'
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', 8000))
    WORKERS: int = int(os.getenv('WORKERS', 1))

    PROJECT: str = os.getenv('PROJECT_NAME', 'Multi-Stage Multi-Label Chest X-Ray Classification')
    DATASET: str = os.getenv('DATASET', 'CheXpert')
    PATHOLOGY_LABELS: List[str] = sorted(
        ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
         'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
         'Fracture', 'Support Devices', 'No Finding']  # The CheXpert dataset Original Labels
    )
    IS_UNCERTAIN_LABEL = os.getenv('IS_UNCERTAIN_LABEL', True)
    UNCERTAIN_LABEL = os.getenv('UNCERTAIN_LABEL', 'Uncertain')
    IS_HIERARCHICAL_TRAINING = os.getenv('IS_HIERARCHICAL_TRAINING', True)
    HIGH_LEVEL_LABELS = sorted(['Abnormal', 'Fluid Accumulation', 'Missing Lung Tissue', 'Other', 'Cardiac', 'Opacity'])
    HIERARCHICAL_LABELS = os.getenv('HIERARCHICAL_LABELS', HIGH_LEVEL_LABELS)
    HIERARCHY_PENALTY = os.getenv('HIERARCHY_PENALTY', 0.5)
    USE_COMPUTED_PENALTIES = os.getenv('USE_COMPUTED_PENALTIES', True)

    # Log directory (absolute path)
    LOG_DIR: str = os.path.abspath(os.getenv('LOG_DIR', './logs'))
    CXR_APP_LOG_DIR = os.path.join(LOG_DIR, 'logfiles', 'cxr_app')
    TRAIN_LOG_DIR = os.path.join(LOG_DIR, 'logfiles', 'train')

    # Model directory and model name
    MODEL_DIR: str = os.path.abspath(os.getenv('MODEL_DIR', './src/assets/models/'))
    MODEL_NAME: str = os.getenv('MODEL_NAME', 'default_model.keras')  # Wed_02_Oct_2024_0814 best Loss
    MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_NAME)
    MONTE_CARLO_UNCERTAINTY_ITERATIONS: int = 10
    NEW_MODEL_PATH: str = os.path.abspath(os.getenv('NEW_MODELS_PATH', f'./fresh_models/{START_TIME}'))

    # Checkpoint path
    IS_CHECKPOINT: bool = os.getenv('IS_CHECKPOINT', 'False').lower() == 'true'
    CKPT_DIR: str = os.path.abspath(os.getenv('CKPT_DIR', './src/assets/models/checkpoints/'))
    CKPT_NAME: str = os.getenv('CKPT_NAME', 'default_chkpt.keras')  # Wed_02_Oct_2024_0814 best AUC
    CKPT_PATH: str = os.path.join(CKPT_DIR, CKPT_NAME)
    NEW_CHKPT_PATH: str = os.path.abspath(os.getenv('NEW_CHKPT_PATH', f'{NEW_MODEL_PATH}/checkpoints/'))

    # Training data settings
    DO_SUBSAMPLE: bool = os.getenv('DO_SUBSAMPLE', 'False').lower() == 'true'
    DO_UPSAMPLE: bool = os.getenv('DO_UPSAMPLE', 'False').lower() == 'true'
    ONLY_PA: bool = os.getenv('ONLY_PA', 'False').lower() == 'true'
    ONLY_M: bool = os.getenv('ONLY_M', 'False').lower() == 'true'
    ONLY_F: bool = os.getenv('ONLY_F', 'False').lower() == 'true'
    TOTAL_NUM_LABELS: int = (len(PATHOLOGY_LABELS) + len(HIERARCHICAL_LABELS)) if IS_HIERARCHICAL_TRAINING else len(PATHOLOGY_LABELS)
    NUM_CLASSES: int = int(os.getenv('NUM_CLASSES', TOTAL_NUM_LABELS))

    # Hyperparameters and other settings
    DO_AUGMENT: bool = os.getenv('DO_AUGMENT', 'False').lower() == 'true'
    DO_PREPROCESS: bool = os.getenv('DO_PREPROCESS', 'False').lower() == 'true'

    IS_TRAINING: bool = os.getenv('IS_TRAINING', 'False').lower() == 'true'
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', 16))
    EPOCHS: int = int(os.getenv('EPOCHS', 100))
    FINE_TUNE_EPOCHS: int = int(os.getenv('FINE_TUNE_EPOCHS', 50))
    TOTAL_EPOCHS: int = EPOCHS + FINE_TUNE_EPOCHS
    LEARNING_RATE: float = float(os.getenv('LEARNING_RATE', 1e-4))
    FINE_LEARNING_RATE: float = float(os.getenv('FINE_LEARNING_RATE', 1e-5))
    FINE_TUNE_AT: int = int(os.getenv('FINE_TUNE_AT', 0))

    INIT_WEIGHTS: str = os.getenv('INIT_WEIGHTS', None)  # None or ImageNet
    IMAGE_SIZE: Tuple[int, int] = (320, 320) if INIT_WEIGHTS is None else (224, 224)
    CHANNELS: int = 3
    INPUT_SHAPE: Tuple[int, int, int] = IMAGE_SIZE + (CHANNELS,)
    INPUT_TENSOR: Any = Input(shape=INPUT_SHAPE)
    USE_CLASS_WEIGHTS: bool = False

    if IS_TRAINING:
        DEVICE = os.getenv('DEVICE', 'GPU')
    BASE_MODEL = DenseNet121(weights=INIT_WEIGHTS,
                             include_top=False,
                             input_tensor=INPUT_TENSOR,
                             input_shape=INPUT_SHAPE)
    # BASE_MODEL: Dict[str, Any] = {
    #     'densenet121': DenseNet121(weights=INIT_WEIGHTS,
    #                                include_top=False,
    #                                input_tensor=INPUT_TENSOR,
    #                                input_shape=INPUT_SHAPE),
    #
    #     'mobilenetv2': MobileNetV2(weights=INIT_WEIGHTS,
    #                                include_top=False,
    #                                input_tensor=INPUT_TENSOR,
    #                                input_shape=INPUT_SHAPE),
    #
    #     'resnet50': ResNet50(weights=INIT_WEIGHTS,
    #                          include_top=False,
    #                          input_tensor=INPUT_TENSOR,
    #                          input_shape=INPUT_SHAPE),
    #
    #     'vgg19': VGG19(weights=INIT_WEIGHTS,
    #                    include_top=False,
    #                    input_tensor=INPUT_TENSOR,
    #                    input_shape=INPUT_SHAPE),
    #
    #     'efficientnetb0': EfficientNetB0(weights=INIT_WEIGHTS,
    #                                      include_top=False,
    #                                      input_tensor=INPUT_TENSOR,
    #                                  input_shape=INPUT_SHAPE)
    # }

    IS_BASE_MODEL_PREPROCESS = os.getenv('BASE_MODEL_PREPROCESS', 'False').lower() == 'true'
    BASE_MODEL_PREPROCESS_UNIT = densenet_preprocess_input
    # BASE_MODEL_PREPROCESS_UNIT: Dict[str, Any] = {
    # 'densenet121': densenet_preprocess_input,
    # 'mobilenetv2': mobilenet_preprocess_input,
    # 'resnet50': resnet50_preprocess_input,
    # 'vgg19': vgg19_preprocess_input,
    # 'efficientnetb0': efficientnet_preprocess_input
    # }

    # Data directory and CSV path
    DATA_DIR: str = os.path.abspath(os.getenv('DATA_DIR', 'data'))
    CSV_PATH: str = os.path.join(DATA_DIR, 'train_visualCheXbert.csv')
    VAL_CSV_PATH: str = os.path.join(DATA_DIR, 'val_labels.csv')

    TEST_DATA_DIR: str = os.path.abspath(os.getenv('DATA_DIR', 'data'))
    TEST_CSV_PATH: str = os.path.join(DATA_DIR, 'test_labels.csv')

    # Validate important paths
    if IS_TRAINING:
        log_path_exists: bool = os.path.exists(LOG_DIR)
        data_path_exists: bool = os.path.exists(DATA_DIR)
        csv_path_exists: bool = os.path.exists(CSV_PATH)
        if not log_path_exists:
            raise FileNotFoundError(f'Log directory {LOG_DIR} does not exist')

        if not data_path_exists:
            raise FileNotFoundError(f'Data directory {DATA_DIR} does not exist')

        if not csv_path_exists:
            raise FileNotFoundError(f'CSV file {CSV_PATH} does not exist')
    else:
        model_path_exists: bool = os.path.exists(MODEL_DIR) or os.path.exists(CKPT_PATH)
        model_file_exists: bool = os.path.isfile(MODEL_PATH) or os.path.isfile(CKPT_PATH)
        if not model_path_exists:
            raise FileNotFoundError(
                f'Neither the model directory ({MODEL_DIR}) nor the checkpoint directory ({CKPT_PATH}) exists'
            )
        if not model_file_exists:
            raise FileNotFoundError(
                f'Neither the model file ({MODEL_PATH}) nor the checkpoint file ({CKPT_PATH}) exists'
            )
