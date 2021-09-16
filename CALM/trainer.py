import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    BatchEncoding,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

from dataset import Batch, SummarizationDataset, CSQADataset, PIQADataset, ANLIDataset, OBQADataset, KILTFEVERDataset, KILTT2TDataset

def get_dataset(tokenizer, type_path, args):
    print(args.data_dir)
    data_dir_leaf = args.data_dir.split("/")[-1]

    if data_dir_leaf == 'commongen':
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_source_length, max_target_length=args.max_target_length)
    elif data_dir_leaf == "c2s" or data_dir_leaf == "cor" or data_dir_leaf == "mix":
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,max_source_length=args.max_seq_length, max_target_length=args.max_seq_length)
    elif data_dir_leaf == 'option1': # choice of string
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_seq_length, max_target_length=2)
    elif data_dir_leaf == 'option2': # string of choice
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_seq_length, max_target_length=int(args.max_seq_length / 2))
    elif data_dir_leaf == 'option3': # True / False
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_seq_length, max_target_length=2)

    elif data_dir_leaf == 'csqa':
        return CSQADataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)
    elif data_dir_leaf == 'piqa':
        return PIQADataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)
    elif data_dir_leaf == "anli":
        return ANLIDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)
    elif data_dir_leaf == "openbookqa":
        return OBQADataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length, use_KB=args.use_KB)

    # KILT Tasks
    elif data_dir_leaf == "kilt_fever":
        return KILTFEVERDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)
    elif data_dir_leaf == "kilt_natural_qa" or data_dir_leaf == "kilt_ay2" or data_dir_leaf == "kilt_trivia_qa":
        return KILTT2TDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_source_length, max_target_length=args.max_target_length, createMultipleSamples=args.expandSamples)


class T5FineTuner(pl.LightningModule):  # type: ignore[misc]
    def __init__(self, hparams: Union[Dict[str, Any], argparse.Namespace]) -> None:
        super(T5FineTuner, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        print("Model params: ", self.hparams)
        self.root_path = Path(__file__).parent.absolute()
        self.model = T5ForConditionalGeneration.from_pretrained(
            hparams.model_name_or_path, cache_dir=self.root_path / "model_cache"
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            hparams.tokenizer_name_or_path, cache_dir=self.root_path / "model_cache"
        )

        if self.hparams.model_parallel:
            if "base" in hparams.model_name_or_path:
                self.device_map = {
                    0: [0, 1, 2],
                    1: [3, 4, 5],
                    2: [6, 7, 8],
                    3: [9, 10, 11],
                }
            else:
                self.device_map = {
                    4: [0, 1, 2, 3, 4, 5],
                    5: [6, 7, 8, 9, 10, 11],
                    6: [12, 13, 14, 15, 16, 17],
                    7: [18, 19, 20, 21, 22, 23],
                }
            self.model.parallelize(self.device_map)

    def is_logger(self):
        return True #temporary fix (only work at single GPU env)

    def forward(
        self,
        input_ids: BatchEncoding,
        attention_mask: BatchEncoding = None,
        decoder_input_ids: BatchEncoding = None,
        decoder_attention_mask: BatchEncoding = None,
        labels: BatchEncoding = None,
    ) -> Any:
        # pylint: disable=arguments-differ
        return self.model(
            input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask.to(self.model.device),
            labels=labels.to(self.model.device),
        )

    def _step(self, batch: Mapping[str, Any]) -> Any:
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch["target_mask"],
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Mapping[str, Any]:
        # pylint: disable=arguments-differ,unused-argument
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {
            "avg_train_loss": avg_train_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> Mapping[str, Any]:
        # pylint: disable=arguments-differ,unused-argument
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def configure_optimizers(self) -> Sequence[Optimizer]:
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        self.opt = optimizer  # pylint: disable=attribute-defined-outside-init
        return [optimizer]

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Optimizer,
            optimizer_idx: int,
            second_order_closure: Optional[Callable] = None,  # type: ignore[type-arg]
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False,
    ) -> None:
        optimizer.step(second_order_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self) -> Mapping[str, Any]:
        tqdm_dict = {
            "loss": "{:.3f}".format(self.trainer.avg_loss),
            "lr": self.lr_scheduler.get_last_lr()[-1],
        }

        return tqdm_dict

    def train_dataloader(self) -> DataLoader[Batch]:
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=0,
        )

        t_total = (
                (
                        len(dataloader.dataset)
                        // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu))
                )
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )

        self.lr_scheduler = scheduler  # pylint: disable=attribute-defined-outside-init
        return dataloader

    def val_dataloader(self) -> DataLoader[Batch]:
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=0)
