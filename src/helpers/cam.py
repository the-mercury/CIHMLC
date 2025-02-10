import os
from typing import Dict, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model as KerasModel
from keras import layers as keras_layers
from numpy import ndarray

from src.config import Config
from src.helpers.model_utils import ModelUtils


class Cam:
    """
    Cam class for generating Class Activation Maps for visualizing model predictions.

    This class helps in understanding which parts of the input image are most important for the model's predictions.
    """

    def __init__(self, model: KerasModel, predictions: Dict[str, float], uncertainties: Dict[str, float],
                 layer_name: str = None):
        """
        Initialize the Cam instance.

        Parameters:
        model (keras.Model): The trained Keras model.
        predictions (Dict[str, float]): The model's predictions.
        layer_name (str, optional): The name of the target convolutional layer. Defaults to None.
        """
        self.model = model
        self.predictions = predictions
        self.uncertainties = uncertainties
        self.layer_name = layer_name if layer_name and self.is_valid_layer_name(
            layer_name) else self.find_target_layer()

    def is_valid_layer_name(self, layer_name: str) -> bool:
        """
        Check if the provided layer name is valid.

        Parameters:
        layer_name (str): The name of the layer to validate.

        Returns:
        bool: True if the layer name is valid, False otherwise.
        """
        try:
            self.model.get_layer(layer_name)
            return True
        except ValueError:
            return False

    def find_target_layer(self) -> str:
        """
        Find the target convolutional layer in the model.

        Returns:
        str: The name of the target convolutional layer.

        Raises:
        ValueError: If no convolutional layers are found in the model.
        """
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras_layers.Conv2D):
                return layer.name
        raise ValueError('No convolutional layers found.')

    def make_cam(self, img_array: Any, clip_heatmap: float = None, clip_color_map: float = None) -> Dict[str, ndarray]:
        """
        Generate Grad-CAM heatmaps for the model's predictions.

        Parameters:
        img_array (Any): The input image array.

        Returns:
        Dict[str, tf.Tensor]: A dictionary of heatmaps for each prediction.
        """
        superimposed_images = {}
        colormap = 'hot_r'
        dynamic_alpha = False
        fig_save_dir = f'{Config.LOG_DIR}/heatmaps/CAM_ch_{clip_heatmap}_cc_{clip_color_map}_cmap_{colormap}_da_{dynamic_alpha}'
        os.makedirs(fig_save_dir, exist_ok=True)
        cam_model = KerasModel(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output[0]]
        )
        for i, key in enumerate(self.predictions.keys()):
            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = cam_model(img_array)
                class_pred = preds[:, i]
            grads = tape.gradient(class_pred, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = tf.where(heatmap > clip_heatmap, heatmap, 0) if clip_heatmap else heatmap

            plt.figure()
            alpha = round((self.predictions[key] * 0.6) + 0.3, 1) if dynamic_alpha else 0.7
            # heatmap = np.multiply(heatmap, self.predictions[key])
            superimposed_image = ModelUtils.overlay_heatmap(heatmap, img_array[0], clip_threshold=clip_color_map,
                                                            cmap=colormap, alpha=alpha)
            superimposed_images[key] = superimposed_image
            fig_id = f'{key}_gradcam_activation_alpha_{alpha}.png' if dynamic_alpha else f'{key}_cam.png'
            h_min, h_max = np.min(heatmap), np.max(heatmap)
            # unique_values = np.unique(np.round(heatmap, 2))
            # boundaries = np.sort(unique_values)
            if clip_color_map is not None:
                n_intervals = int((1 - clip_color_map) * 10)
                boundaries = np.round(np.linspace(h_min, h_max, n_intervals + 1), 1)
                norm = mcolors.BoundaryNorm(boundaries, ncolors=256)
                im = plt.imshow(superimposed_image, cmap=colormap, norm=norm)
                cbar = plt.colorbar(im, label='Heatmap Intensity', boundaries=boundaries, ticks=boundaries)
                cbar.ax.set_yticklabels([b for b in boundaries])
            else:
                im = plt.imshow(superimposed_image, cmap=colormap, vmin=h_min, vmax=h_max)
                plt.colorbar(im, label='Heatmap Intensity')

            # # Alpha transparency bar
            # alpha_min, alpha_max = 0.3, 0.9  # Alpha range based on predictions
            # alpha_gradient = np.linspace(alpha_min, alpha_max, 256).reshape(1, -1)
            # alpha_cmap = colormap
            # alpha_mappable = plt.cm.ScalarMappable(cmap=alpha_cmap)
            # alpha_mappable.set_array(alpha_gradient)
            # alpha_norm = mcolors.Normalize(vmin=alpha_min, vmax=alpha_max)
            #
            # tick_values = np.linspace(alpha_min, alpha_max, 6)
            # cbar_alpha = plt.colorbar(alpha_mappable, ax=plt.gca(), orientation='horizontal')
            # cbar_alpha.locator = FixedLocator(tick_values)
            # cbar_alpha.update_ticks()
            # cbar_alpha.ax.set_xticklabels([f'{int(a * 100)}%' for a in tick_values])
            # cbar_alpha.set_label("Transparency Level", fontsize=10)

            plt.title(f'{key}: {self.predictions[key]:.2f}±{self.uncertainties[key]:.2f}')
            plt.axis('off')
            plt.savefig(os.path.join(fig_save_dir, fig_id), bbox_inches='tight')
            plt.close()

        return superimposed_images
