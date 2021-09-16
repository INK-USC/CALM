import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from dataset import Batch, SummarizationDataset
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union
from trainer import T5FineTuner
from torch.optim import Optimizer
import argparse
import glob, os, re
from utils import get_best_checkpoint
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

def get_dataset(tokenizer, type_path, args):
    print(args.data_dir)
    data_dir_leaf = args.data_dir.split("/")[-1]
    if data_dir_leaf == 'option1': # choice of string
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_seq_length, max_target_length=2)
    if data_dir_leaf == 'option2': # string of choice
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_seq_length, max_target_length=int(args.max_seq_length / 2))
    if data_dir_leaf == 'option3': # True / False
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_seq_length, max_target_length=2)

class T5GANFineTuner(pl.LightningModule):
    def __init__(self, hparams: Union[Dict[str, Any], argparse.Namespace]) -> None:
        super(T5GANFineTuner, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        print("Model params: ", self.hparams)
        self.root_path = Path(__file__).parent.absolute()
        self.data_dir_leaf = self.hparams.data_dir.split("/")[-1]
        self.tokenizer = T5Tokenizer.from_pretrained(
            hparams.tokenizer_name_or_path, cache_dir=self.root_path / "model_cache"
        )
        if hparams.checkpoint_dir != '':
            best_checkpoint_path = get_best_checkpoint(self.hparams.checkpoint_dir)
            print("Using checkpoint = ", str(best_checkpoint_path))
            checkpoint_model = T5FineTuner.load_from_checkpoint(best_checkpoint_path)
            self.model = checkpoint_model.model
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                hparams.model_name_or_path, cache_dir=self.root_path / "model_cache"
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

        self.generator_max_length = 128
        self.generator_min_length = 1

        self.generator_weight = 1.0
        self.discriminator_weight = 1.0


    def is_logger(self):
        return True

    def forward(self, discriminator_input_ids,
                discriminator_attention_mask=None,
                discriminator_decoder_input_ids=None,
                discriminator_decoder_attention_mask=None,
                discriminator_labels=None):

        # original sentence / targets
        device = self.model.device
        batch_sentences = self.tokenizer.batch_decode(discriminator_input_ids)
        discriminator_labels[discriminator_labels[:, :] == -100] = self.tokenizer.pad_token_id
        batch_labels = self.tokenizer.batch_decode(discriminator_labels)

        # which setnence is correct ? option 1 : original 2 : concept-shuffled 1
        # extract sentence that we gonna feed into the generator
        batch_placeholder = []
        deleted_idx = []
        if self.data_dir_leaf == 'option1':
            sentence_prefix = "Which sentence is correct?: "
            without_prefix = [sent.split(sentence_prefix)[1] for sent in batch_sentences]
            for batch_idx, (without_prefix_sentence, b_label) in enumerate(zip(without_prefix, batch_labels)):
                after_option1 = without_prefix_sentence.split('options: 1: ')[1]
                after_option2 = after_option1.split('2: ')
                option = [after_option2[0].strip(), after_option2[1].strip()]
                if b_label == '1':
                    batch_placeholder.append([option, batch_idx, 2]) # need to change second one
                elif b_label == '2':
                    batch_placeholder.append([option, batch_idx, 1]) # need to change first one
                else:
                    deleted_idx.append(batch_idx)

        if self.data_dir_leaf == 'option2':
            sentence_prefix = "Which sentence is correct?: "
            without_prefix = [sent.split(sentence_prefix)[1] for sent in batch_sentences]
            for batch_idx, (without_prefix_sentence, b_label) in enumerate(zip(without_prefix, batch_labels)):
                if len(without_prefix_sentence.split('options: 1: ')) != 2:
                    deleted_idx.append(batch_idx)
                    continue
                after_option1 = without_prefix_sentence.split('options: 1: ')[1]
                if len(after_option1.split('2: ')) != 2:
                    deleted_idx.append(batch_idx)
                    continue
                after_option2 = after_option1.split('2: ')
                option = [after_option2[0].strip(), after_option2[1].split('</s>')[0].strip()]
                if option[0] == b_label.split('</s>')[0].strip():
                    batch_placeholder.append([option, batch_idx, 2]) # need to change second one
                elif option[1] == b_label.split('</s>')[0].strip():
                    batch_placeholder.append([option, batch_idx, 1]) # need to change first one
                else:
                    deleted_idx.append(batch_idx) # trimmed

        if self.data_dir_leaf == 'option3':
            sentence_prefix = "Does this sentence make sense?: "
            without_prefix = [sent.split(sentence_prefix)[1] for sent in batch_sentences]
            for batch_idx, (option, b_label) in enumerate(zip(without_prefix, batch_labels)):
                if b_label == 'true':
                    batch_placeholder.append([option, batch_idx, "true"])
                if b_label == 'false':
                    batch_placeholder.append([option, batch_idx, "false"])

        generator_prefix = "correct the following sentence : "
        fake_source = []
        fake_target = []
        for fake_batch in batch_placeholder:
            if fake_batch[2] == 1:
                fake_source.append(generator_prefix + fake_batch[0][0] + " </s>")
                fake_target.append(fake_batch[0][1] + " </s>")
            if fake_batch[2] == 2:
                fake_source.append(generator_prefix + fake_batch[0][1] + " </s>")
                fake_target.append(fake_batch[0][0] + " </s>")
            if fake_batch[2] == "false":
                #TODO : difficult to get original sentence in this setting
                fake_source.append(generator_prefix + fake_batch[0] + " </s>")
                fake_target.append(generator_prefix + fake_batch[0] + " </s>")


        # using source and target to train the generator
        generator_input = self.tokenizer.batch_encode_plus(
            fake_source, max_length=self.generator_max_length, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        generator_target = self.tokenizer.batch_encode_plus(
            fake_target, max_length=self.generator_max_length, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        generator_labels = generator_target["input_ids"]
        generator_labels[generator_labels[:, :] == self.tokenizer.pad_token_id] = -100
        generator_outputs = self.model(generator_input["input_ids"].to(device),
                              attention_mask=generator_input["attention_mask"].to(device),
                              decoder_input_ids=None,
                              decoder_attention_mask=generator_target["attention_mask"].to(device),
                              labels=generator_labels.to(device))

        generator_loss = generator_outputs[0]

        fake_sentences_input_ids = self.model.generate(
            input_ids=generator_input["input_ids"].to(device),
            attention_mask=generator_input["attention_mask"].to(device),
            num_beams=5,
            length_penalty=0.6,
            max_length=self.generator_max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
            min_length=self.generator_min_length + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        fake_sentences = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in fake_sentences_input_ids]

        discriminator_input_sentences = []
        for choice_list, fake_sentence in zip(batch_placeholder, fake_sentences):
            if choice_list[2] == 1: # if the first one is changed
                discriminator_input = sentence_prefix + 'options: 1: ' + fake_sentence + ' 2: ' + choice_list[0][1] + ' </s>'
            if choice_list[2] == 2: # if the second one is changed
                discriminator_input = sentence_prefix + 'options: 1: ' + choice_list[0][0] + ' 2: ' + fake_sentence + ' </s>'
            discriminator_input_sentences.append(discriminator_input)

        discriminator_input_regenerated = self.tokenizer.batch_encode_plus(
            discriminator_input_sentences, max_length=self.hparams.max_seq_length, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        delete_mask = [1] * len(batch_sentences)
        for idx in deleted_idx:
            delete_mask[idx] = 0
        delete_mask_tensor = torch.ByteTensor(delete_mask)

        discriminator_labels = discriminator_labels[delete_mask_tensor]
        discriminator_decoder_attention_mask = discriminator_decoder_attention_mask[delete_mask_tensor]
        discriminator_output = self.model(discriminator_input_regenerated["input_ids"].to(device),
                                          attention_mask=discriminator_input_regenerated["attention_mask"].to(device),
                                          decoder_input_ids=discriminator_decoder_input_ids,
                                          decoder_attention_mask=discriminator_decoder_attention_mask.to(device),
                                          labels=discriminator_labels.to(device))

        discriminator_loss = discriminator_output[0]

        return self.generator_weight * generator_loss + self.discriminator_weight * discriminator_loss

    def _step(self, batch: Mapping[str, Any]) -> Any:
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            discriminator_input_ids=batch["source_ids"],
            discriminator_attention_mask=batch["source_mask"],
            discriminator_labels=labels,
            discriminator_decoder_attention_mask=batch['target_mask'],
        )

        loss = outputs

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
