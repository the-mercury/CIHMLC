from typing import Dict, Any, Tuple
import logging

# import cv2 as cv
import numpy as np

from src.config import Config
from src.helpers.data_utils import DataUtils
from src.helpers.cam import Cam
from src.helpers.model_utils import ModelUtils

logger = logging.getLogger('XRayInference')


class XRayInference:
    """
    XRayInference class for running inference on X-ray images using a trained model.

    This class handles preprocessing, prediction, uncertainty estimation, and postprocessing for generating CAM heatmaps.
    """

    def __init__(self, model: Any):
        """
        Initialize the XRayInference instance.

        Parameters:
        model (Any): The trained model to use for inference.
        """
        super().__init__()
        self.__model = model
        self.input_size = Config.IMAGE_SIZE
        self.pathologies = Config.PATHOLOGY_LABELS
        self.pathologies.append(Config.UNCERTAIN_LABEL)
        self.pathologies.extend(Config.HIERARCHICAL_LABELS)
        # self.transform = cv.createCLAHE(clipLimit=1, tileGridSize=(10, 10))
        self.transform = None
        logger.info(f'XRayInference initialized with these labels:\n{self.pathologies}')

    async def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image.

        Parameters:
        image (np.ndarray): The input image array.

        Returns:
        np.ndarray: The preprocessed image array with an added batch dimension.
        """
        preprocessed_image = DataUtils.preprocess_image(image=image, transform=self.transform)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        if Config.IS_BASE_MODEL_PREPROCESS:
            preprocess_unit = ModelUtils.get_preprocess_unit(base_model=None)
            preprocessed_image = preprocess_unit(preprocessed_image)
        return preprocessed_image

    async def _predict(self, preprocessed_image: np.ndarray) -> Dict[str, float]:
        """
        Predict the pathology scores for the preprocessed image.

        Parameters:
        preprocessed_image (np.ndarray): The preprocessed image array.

        Returns:
        Dict[str, float]: The predicted scores for each pathology.
        """
        scores = self.__model.predict(preprocessed_image).flatten().tolist()
        predictions = dict(zip(self.pathologies, scores))
        return predictions

    async def _predict_with_uncertainty(self, image: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Predict pathology scores with uncertainty estimation using Monte Carlo dropout.

        Parameters:
        image (np.ndarray): The input image array.

        Returns:
        Tuple[Dict[str, float], Dict[str, float]]: The predicted scores and uncertainties for each pathology.
        """
        raw_scores = [self.__model(image, training=True) for _ in range(Config.MONTE_CARLO_UNCERTAINTY_ITERATIONS)]
        scores = np.array(raw_scores).mean(axis=0).flatten().tolist()
        intervals = np.array(raw_scores).std(axis=0).flatten().tolist()
        predictions = dict(zip(self.pathologies, scores))
        uncertainties = dict(zip(self.pathologies, intervals))
        return predictions, uncertainties

    async def _postprocess(self, predictions: Dict[str, float], uncertainties: Dict[str, float], image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps for the predictions.

        Parameters:
        predictions (Dict[str, float]): The predicted scores for each pathology.
        image (np.ndarray): The input image array.

        Returns:
        Dict[str, np.ndarray]: The heatmaps for each pathology.
        """
        visualize = Cam(self.__model, predictions, uncertainties)
        superimposed_images = visualize.make_cam(image, clip_heatmap=0.5, clip_color_map=0.5)

        return superimposed_images

    async def run_inference(self, image_base64: str) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float]]:
        """
        Run inference on the input image.

        Parameters:
        cxr_base64 (str): The input image encoded in base64 format.

        Returns:
        Tuple[Dict[str, str], Dict[str, float], Dict[str, float]]: The heatmaps in base64, predictions, and uncertainties.
        """
        superimposed_images_base64 = {}
        image_array = DataUtils.decode_base64_to_image(image_base64)
        preprocessed_image = await self._preprocess_image(image_array)
        predictions, uncertainties = await self._predict_with_uncertainty(preprocessed_image)
        superimposed_images = await self._postprocess(predictions, uncertainties, preprocessed_image)
        for key in superimposed_images:
            superimposed_images_base64[key] = DataUtils.encode_image_to_base64(superimposed_images[key])
        return superimposed_images_base64, predictions, uncertainties
