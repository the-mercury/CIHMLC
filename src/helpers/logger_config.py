import logging
import os

from keras.callbacks import Callback
from keras.utils import Progbar

from src.config import Config


class LoggerConfig:
    """
    LoggerConfig class provides a static method to configure and get a logger
    with a specific name and log directory. This helps to centralize the logging
    configuration across the project.
    """

    @staticmethod
    def get_logger(name: str, log_dir: str) -> logging.Logger:
        """
        Configures and returns a logger with the specified name and log directory.

        Parameters:
        name : str
            The name of the logger.
        log_dir : str
            The directory where log files will be stored.

        Returns:
        logging.Logger
            The configured logger.
        """
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Clear existing log handlers to avoid duplication
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] [%(name)s] line %(lineno)d: %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/{name}_{Config.START_TIME}.log'),
                logging.StreamHandler()
            ]
        )

        return logging.getLogger(name)


class LoggingCallback(Callback):
    """
    Callback that logs messages at the end of each epoch, mimicking Keras's format.
    """

    def __init__(self, logger: callable):
        """
        Initialize the LoggingCallback.

        Parameters:
        logger (callable): Logger function to log messages.
        """
        super().__init__()
        self.logger = logger
        self.progbar = None
        self.target = None

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        """
        Called at the beginning of each epoch.

        Parameters:
        logs (dict, optional): Dictionary of logs.
        """
        self.target = self.params.get('samples') or self.params.get('steps')
        if self.target is None:
            raise ValueError('LoggingCallback requires "samples" or "steps" in params.')

        self.progbar = Progbar(target=self.target, verbose=1, stateful_metrics=self.params.get('metrics', []))
        # self.logger(f'Running epoch {epoch + 1}/{self.params["epochs"]}...')

    def on_batch_end(self, batch: int, logs: dict = None) -> None:
        """
        Called at the end of each batch.

        Parameters:
        batch (int): The current batch number.
        logs (dict, optional): Dictionary of logs.
        """
        logs = logs or {}
        self.progbar.update(batch + 1, values=[(k, logs[k]) for k in self.params.get('metrics', []) if k in logs])

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """
        Called at the end of each epoch.

        Parameters:
        epoch (int): The current epoch number.
        logs (dict, optional): Dictionary of logs.
        """
        logs = logs or {}
        self.progbar.update(self.target, values=[(k, logs[k]) for k in self.params.get('metrics', []) if k in logs])
        msg = f'Epoch {epoch + 1}/{self.params["epochs"]}: ' + ', '.join(f'{k}: {v:.4e}' for k, v in logs.items())
        self.logger(msg)
