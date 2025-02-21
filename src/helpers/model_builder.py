import logging
from typing import Optional, Dict, List

from keras import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, BatchNormalization
from keras.metrics import BinaryAccuracy, AUC, F1Score
from keras.optimizers import Adam
from pandas import DataFrame as PandasDataFrame

from src.config import Config
from src.hierarchical_binary_cross_entropy import HierarchicalBinaryCrossEntropy

logger = logging.getLogger('ModelBuilder')


class ModelBuilder:
    """
    A class for building and compiling hierarchical classification model based on configurable settings.

    This class manages the initialization, configuration, and compilation of models tailored for with support for
    hierarchical labels and customizable architectural choices.

    Attributes:
        __parent_indices (List[int]): List of parent label indices in the hierarchical label structure.
        __child_indices (List[List[int]]): List of lists containing child label indices for each parent.
        __penalty_type (str): String indicating the type of penalty used ('fixed_{value}' or 'data_driven').
        __penalties (Optional[Dict[int, Dict[int, float]]]): Penalties for the hierarchical loss function,
            computed based on data.
        __hierarchy_penalty (Optional[float]): Fixed penalty for the hierarchical loss function when not using
            data-driven penalties.
        __penalty_scale (Optional[float]): Scaling factor applied to the penalties.
        __num_classes (int): The number of classes to predict; derived from the config file if not explicitly provided.
        __model (Model): The compiled Keras model.
        __base_model (Model): The base model architecture used for feature extraction.
        __preprocess_unit (Optional[Callable]): Preprocessing function associated with the base model.
        __input_shape (Tuple[int, int, int]): Input shape of the model images.
        __learning_rate (float): Learning rate for the optimizer.
        __input_tensor (Tensor): Input tensor for the model.
        __metrics (List[Metric]): List of metrics to evaluate during training.

    Methods:
        __init__(parent_indices, child_indices, hierarchy_num=None, penalties=None,
                 hierarchy_penalty=None, penalty_scale=None, num_classes=None, base_model=None, use_gap=True):
            Initializes the ModelBuilder with the given parameters.

        __build_model():
            Constructs the neural network architecture based on the specified configurations.

        __compile_model():
            Compiles the model with the appropriate optimizer, loss function, and metrics.

        get_model() -> Model:
            Returns the compiled Keras model.

        get_config() -> PandasDataFrame:
            Provides a DataFrame containing the current model configuration.

    Usage:
        # Initialize a model builder
        model_builder = ModelBuilder(num_classes=21, ...)
        # Retrieve the compiled_model
        compiled_model = model_builder.get_model()
    """

    def __init__(self,
                 parent_indices: List[int], child_indices: List[List[int]],
                 penalties: Optional[Dict[int, Dict[int, float]]] = None,
                 hierarchy_penalty: Optional[float] = None,
                 penalty_scale: Optional[float] = None,
                 num_classes: Optional[int] = None,
                 base_model: Optional[str] = None):
        """
        Initialize the ModelBuilder.

        Parameters:
            parent_indices (List[int]): List of parent label indices in the hierarchical label structure.
            child_indices (List[List[int]]): List of lists containing child label indices for each parent.
            penalties (Optional[Dict[int, Dict[int, float]]]): Penalties for the hierarchical loss function.
            hierarchy_penalty (Optional[float]): Fixed penalty for the hierarchical loss function when not using data-driven penalties.
            penalty_scale (Optional[float]): Scaling factor applied to the penalties.
            num_classes (Optional[int]): The number of classes to predict.
            base_model (Optional[str]): Identifier for the base model architecture.
        """
        logger.info('Creating a new ModelBuilder instance')
        self.__model = None
        self.__input_shape = Config.INPUT_SHAPE
        self.__learning_rate = Config.LEARNING_RATE
        self.__input_tensor = Config.INPUT_TENSOR
        self.__parent_indices = parent_indices
        self.__child_indices = child_indices
        self.__penalties = penalties
        self.__penalty_scale = penalty_scale
        self.__hierarchy_penalty = hierarchy_penalty
        self.__num_classes = num_classes if num_classes else Config.NUM_CLASSES
        self.__base_model = base_model if base_model else Config.BASE_MODEL
        self.__preprocess_unit = Config.BASE_MODEL_PREPROCESS_UNIT if Config.IS_BASE_MODEL_PREPROCESS else None
        self.__base_model.trainable = True
        logger.info(f'ModelBuilder initialized with the base model: {self.__base_model}, '
                    f'initial weights: {Config.INIT_WEIGHTS}, '
                    f'preprocess unit: {self.__preprocess_unit}, '
                    f'trainable: {self.__base_model.trainable}')
        logger.info(f'ModelBuilder penalty_scale: {self.__penalty_scale}')
        self.__penalty_type = f'fixed_{self.__hierarchy_penalty}' if self.__hierarchy_penalty is not None else f'data_driven'
        self.__metrics = None

    def __build_model(self):
        """
        Build the model architecture.

        Constructs the neural network by adding custom layers on top of the base model,
        including optional convolutional and pooling layers, followed by dense layers.
        """
        x = Conv2D(512, (3, 3), padding='same', activation='relu', name='top_conv2d')(self.__base_model.output)
        x = BatchNormalization(name='top_bn')(x)
        x = GlobalAveragePooling2D(name='top_gap2d')(x)
        x = Dense(128, activation='relu', name='top_dense')(x)
        x = Dropout(0.5, name='top_dropout')(x)
        outputs = Dense(self.__num_classes, activation='sigmoid', name='top_pred')(x)
        self.__model = Model(inputs=self.__input_tensor, outputs=outputs)

    def __compile_model(self):
        """
        Compile the model with specified loss and metrics.

        Sets up the optimizer, loss function (including hierarchical penalties), and evaluation metrics.
        """
        self.__metrics = [
            BinaryAccuracy(name='accuracy'),
            F1Score(average='weighted', threshold=0.5, name='f1_score'),
            AUC(multi_label=True, name='AUC')
        ]
        self.__model.compile(
            optimizer=Adam(learning_rate=self.__learning_rate),
            loss=HierarchicalBinaryCrossEntropy(
                parent_indices=self.__parent_indices,
                child_indices=self.__child_indices,
                penalties=self.__penalties,
                hierarchy_penalty=self.__hierarchy_penalty,
                penalty_scale=self.__penalty_scale
            ),
            metrics=self.__metrics
        )

        self.__model.name = (f'{Config.DATASET}_base_{self.__base_model.name}-hierarchical-penalty_'
                             f'{self.__penalty_type}-scale_factor_{self.__penalty_scale}-cls_{self.__num_classes}'
                             f'--{Config.START_TIME}')

    def get_model(self) -> Model:
        """
        Get the compiled model.

        Returns:
            Model: The compiled Keras model ready for training.
        """
        self.__build_model()
        self.__compile_model()
        return self.__model

    def get_config(self) -> PandasDataFrame:
        """
        Get the model configuration.

        Returns:
            PandasDataFrame: A DataFrame containing the model configuration parameters.
        """
        config_dict = {
            'dataset': Config.DATASET,
            'structure': 'Hierarchical',
            'penalty type': self.__penalty_type,
            'penalty scale factor': self.__penalty_scale,
            'model name': self.__model.name,
            'input shape': self.__input_shape,
            'learning rate': self.__learning_rate,
            'epochs': Config.EPOCHS,
            'batch size': Config.BATCH_SIZE,
            'num of classes': self.__num_classes,
            'base_model': self.__base_model.name,
            'initial weights': Config.INIT_WEIGHTS,
            'base model preprocess unit': self.__preprocess_unit,
            'loss function': self.__model.loss,
            'optimizer': self.__model.optimizer,
            'metrics': self.__metrics,
        }
        return PandasDataFrame.from_dict(config_dict, orient='index', columns=[''])
