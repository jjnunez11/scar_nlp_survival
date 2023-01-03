# Could do old one based on
# https://github.com/mabdullah1994/Text-Classification-with-BERT-PyTorch/blob/master/Classifier.py
# Instead just use Pytorch Lightning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
import time
import pandas as pd
import pytorch_lightning as pl


class TransformerTrainer(pl.Trainer):
    def __init__(self, config, checkpoint_callback, logger):

        # Patience; stopping early if prediction not improving
        early_stop_callback = EarlyStopping(monitor="val_bal",
                                            min_delta=0.00,
                                            patience=config.patience,
                                            verbose=False,
                                            mode="max")

        super().__init__(max_epochs=config.epochs,
                         gpus=1,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         progress_bar_refresh_rate=30,
                         logger=logger)

        self.start = time.time()  # Don't want to mess with inherited fit so just grab start time here

    def get_results(self):
        """
        Quick function to return our custom logger's history, as well as the
        start time, to maintain functionality with the rest of the codebase results handling
        :return:
        """

        train_history, dev_history, test_history = self.logger[1].get_history()
        start = self.start

        return train_history, dev_history, test_history, start


class MyLogger(LightningLoggerBase):

    def __init__(
        self,
        save_dir: str,
        # name: Optional[str] = "default",
        # version: Optional[Union[int, str]] = None,
        # prefix: str = "",
    ):
        super().__init__()
        self._save_dir = save_dir
        # self._name = name or ""
        # self._version = version
        # self._prefix = prefix
        # self._experiment = None

        self.train_history = pd.DataFrame()
        self.dev_history = pd.DataFrame()
        self.test_history = pd.DataFrame()

    @property
    def name(self):
        return "MyLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here

        if 'dev_perf' in metrics:
            dev_perf = metrics["dev_perf"]
            dev_perf["epoch"] = metrics["epoch"]
            self.dev_history = self.dev_history.append(dev_perf, ignore_index=True)
            self.dev_history.index.name = "epoch"

        elif 'train_perf' in metrics:
            pass

        elif 'train_perf_epoch' in metrics:
            train_perf = metrics["train_perf_epoch"]
            train_perf["epoch"] = metrics["epoch"]
            self.train_history = self.train_history.append(train_perf, ignore_index=True)
            self.train_history.index.name = "epoch"

        elif 'test_perf' in metrics:
            test_perf = metrics["test_perf"]
            test_perf["epoch"] = metrics["epoch"]
            self.test_history = self.test_history.append(test_perf, ignore_index=True)
            self.test_history.index.name = "epoch"
        else:
            raise ValueError(f"Unexpected metrics, here they are: {metrics}")

        pass

    # @rank_zero_only
    # def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
    #    print('Triggered a save!')
    #    super().save()

    # @rank_zero_only
    # def finalize(self, status):
    #    # Optional. Any code that needs to be run after training
    #    # finishes goes here
    #    print('Finalizing!')
    #    pass

    def get_history(self):
        return self.train_history, self.dev_history, self.test_history
