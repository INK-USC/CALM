import logging
import os

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):  # type: ignore[misc]
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("%s = %s\n", key, str(metrics[key]))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(
                pl_module.hparams.output_dir, "test_results.txt"
            )
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("%s = %s\n", key, str(metrics[key]))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))
