import logging
from typing import List, Dict, Optional

import tensorflow as tf
from keras.saving import register_keras_serializable

logger = logging.getLogger('HierarchicalBinaryCrossEntropy')


@register_keras_serializable(package='Custom')
class HierarchicalBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Custom loss function for hierarchical multi-label classification.
    This loss function combines standard binary cross-entropy loss with additional penalties
    for inconsistent parent-child predictions, where a child label is predicted as positive
    while its parent label is predicted as negative.

    Attributes:
    parent_indices : List[int]
        List of indices for the parent labels in the prediction vector.
    child_indices : List[List[int]]
        List of lists, where each sublist contains the indices of the child labels corresponding to a specific parent.
    penalties : Optional[Dict[int, Dict[int, float]]]
        A nested dictionary containing penalty values for each parent-child pair.
    hierarchy_penalty : float
        A fixed penalty factor applied when a child label is predicted as positive but its parent label is negative.
    penalty_scale : float
        A scaling factor applied to the penalties to adjust their impact relative to the binary cross-entropy loss.
    reduction : str
        The type of reduction to apply to loss. By default, it is 'sum_over_batch_size'. It can also be 'none', or 'sum'.

    Methods:
    call(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor
        Computes the combined binary cross-entropy and hierarchy-aware loss between the true and predicted labels.
    get_config() -> dict:
        Returns the configuration of the loss function for serialization.
    from_config(config: dict) -> HierarchicalBinaryCrossEntropy:
        Recreates the loss function from the configuration.
    """

    def __init__(self,
                 parent_indices: List[int],
                 child_indices: List[List[int]],
                 penalties: Optional[Dict[int, Dict[int, float]]] = None,
                 hierarchy_penalty: float = 0.1,
                 penalty_scale: float = 1.0,
                 reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name: str = "hierarchical_binary_crossentropy"):
        """
        Initializes the HierarchicalBinaryCrossEntropy loss function.

        Parameters:
        parent_indices : List[int]
            List of indices for the parent labels in the prediction vector.
        child_indices : List[List[int]]
            List of lists, where each sublist contains the indices of the child labels corresponding to a specific parent.
        penalties : Optional[Dict[int, Dict[int, float]]], optional
            A nested dictionary containing penalty values for each parent-child pair (default is None).
        hierarchy_penalty : float, optional
            A fixed penalty factor applied when a child label is predicted as positive but its parent label is negative (default is 0.1).
        penalty_scale : float, optional
            A scaling factor applied to the penalties to adjust their impact relative to the binary cross-entropy loss (default is 1.0).
        reduction : str, optional
            Type of reduction to apply to the loss (default is 'sum_over_batch_size'). Can be 'none', 'sum', or 'sum_over_batch_size'.
        name : str, optional
            Name for the custom loss function (default is "hierarchical_binary_crossentropy").
        """
        super(HierarchicalBinaryCrossEntropy, self).__init__(reduction=reduction, name=name)
        self.parent_indices = parent_indices
        self.child_indices = child_indices
        self.penalties = penalties
        self.hierarchy_penalty = hierarchy_penalty
        self.penalty_scale = penalty_scale
        if penalties is not None:
            logger.info(
                f'Initialized HierarchicalBinaryCrossEntropy with computed penalties and penalty_scale: {self.penalty_scale}')
        else:
            logger.info(
                f'Initialized HierarchicalBinaryCrossEntropy with fixed hierarchy_penalty: {self.hierarchy_penalty}')

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the hierarchical binary cross-entropy loss.

        Parameters:
        y_true : tf.Tensor
            Ground truth tensor with shape (batch_size, num_labels), where num_labels includes all parent and child labels.
        y_pred : tf.Tensor
            Predicted probabilities with shape (batch_size, num_labels).

        Returns:
        tf.Tensor
            A tensor representing the combined binary cross-entropy and hierarchical penalty loss.
        """
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        hierarchy_penalty_total = 0.0
        for parent_idx, children in zip(self.parent_indices, self.child_indices):
            parent_pred = y_pred[:, parent_idx]
            for child_idx in children:
                child_pred = y_pred[:, child_idx]
                if self.penalties is not None:
                    penalty_value = self.penalties[parent_idx][child_idx] * self.penalty_scale
                else:
                    penalty_value = self.hierarchy_penalty
                penalty = tf.where(
                    (parent_pred < 0.5) & (child_pred > 0.5),
                    penalty_value * tf.ones_like(child_pred),
                    tf.zeros_like(child_pred)
                )
                hierarchy_penalty_total += penalty
        hbce_loss = bce + hierarchy_penalty_total
        return hbce_loss

    def get_config(self) -> dict:
        """
        Returns the configuration of the loss function for serialization.

        Returns:
        dict
            A dictionary containing the configuration of the loss function.
        """
        config = super(HierarchicalBinaryCrossEntropy, self).get_config()
        config.update({
            'parent_indices': self.parent_indices,
            'child_indices': self.child_indices,
            'penalties': self.penalties,
            'hierarchy_penalty': self.hierarchy_penalty,
            'penalty_scale': self.penalty_scale
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        """
        Recreates the loss function from the configuration.

        Parameters:
        config : dict
            The configuration dictionary returned by `get_config`.

        Returns:
        HierarchicalBinaryCrossEntropy
            The deserialized HierarchicalBinaryCrossEntropy instance.
        """
        parent_indices = config.pop('parent_indices')
        child_indices = config.pop('child_indices')
        penalties = config.pop('penalties')
        hierarchy_penalty = config.pop('hierarchy_penalty')
        penalty_scale = config.pop('penalty_scale')
        return cls(parent_indices=parent_indices,
                   child_indices=child_indices,
                   penalties=penalties,
                   hierarchy_penalty=hierarchy_penalty,
                   penalty_scale=penalty_scale,
                   **config)
