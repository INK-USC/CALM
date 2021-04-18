# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import tensorflow.compat.v1 as tf

import os
import re, string
import json
from typing import Mapping, Sequence

from torch.utils.data import Dataset
from transformers import BatchEncoding

Batch = Mapping[str, Sequence[BatchEncoding]]

class SummarizationDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_source_length=32, max_target_length=32):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        self.inputs = self.encode_file(self.tokenizer, os.path.join(self.data_dir, self.type_path + ".source"),
                                       self.max_source_length)
        self.targets = self.encode_file(self.tokenizer, os.path.join(self.data_dir, self.type_path + ".target"),
                                        self.max_target_length)

    def encode_file(self, tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt"):
        examples = []
        with open(data_path, "r") as f:
            for text in f.readlines():
                tokenized = tokenizer.batch_encode_plus(
                    [text], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
                    truncation=True
                )
                examples.append(tokenized)
        return examples


class InputExample(object):
    """A single multiple choice question. Here "article" is optional"""

    def __init__(self, qid, question, answers, label, article=None):
        """Construct an instance."""
        self.qid = qid
        self.question = question
        self.answers = answers
        self.label = label
        self.article = article


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a JSON file."""
        with tf.gfile.Open(input_file, "r") as f:
            return json.load(f)

    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a JSON Lines file."""
        with tf.gfile.Open(input_file, "r") as f:
            return [json.loads(ln) for ln in f]


class CommonsenseQAProcessor(DataProcessor):
    """Processor for the CommonsenseQA data set."""

    SPLITS = ['qtoken', 'rand']
    LABELS = ['A', 'B', 'C', 'D', 'E']

    TRAIN_FILE_NAME = 'train_{split}_split.jsonl'
    DEV_FILE_NAME = 'dev_{split}_split.jsonl'
    TEST_FILE_NAME = 'test_{split}_split_no_answers.jsonl'

    def __init__(self, split):
        if split not in self.SPLITS:
            raise ValueError('split must be one of {", ".join(self.SPLITS)}.')
        self.split = split

    def get_train_examples(self, data_dir):
        train_file_name = self.TRAIN_FILE_NAME.format(split=self.split)
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, train_file_name)), 'train')

    def get_dev_examples(self, data_dir):
        dev_file_name = self.DEV_FILE_NAME.format(split=self.split)
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, dev_file_name)), 'dev')

    def get_test_examples(self, data_dir):
        test_file_name = self.TEST_FILE_NAME.format(split=self.split)

        return self._create_examples(self._read_jsonl(os.path.join(data_dir, test_file_name)), 'test')

    def get_labels(self):
        return [0, 1, 2, 3, 4]

    def _create_examples(self, lines, set_type):
        examples = []
        for line in lines:
            qid = line['id']
            question = line['question']['stem']
            answers = [choice['text'] for choice in sorted(line['question']['choices'], key=lambda c: c['label'])]
            label = self.LABELS.index(line.get('answerKey', 'A'))
            examples.append(InputExample(
                qid=qid,
                question=question,
                answers=answers,
                label=label))

        return examples


class CSQADataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.proc = CommonsenseQAProcessor('rand')

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        else:
            examples = self.proc.get_dev_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.question
        options = ['%s: %s' % (i, option) for i, option in zip('12345', example.answers)]
        options = " ".join(options)
        input_ = "context: %s  options: %s </s>" % (input_, options)
        target = "%s </s>" % str(int(example.label) + 1)

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)


class PIQAProcessor(DataProcessor):
    """Processor for the PIQA data set."""

    LABELS = ['sol1', 'sol2']

    TRAIN_FILE_NAME = 'train.jsonl'
    TRAIN_LABEL_NAME = 'train-labels.lst'
    DEV_FILE_NAME = 'valid.jsonl'
    DEV_LABEL_NAME = 'valid-labels.lst'
    TEST_FILE_NAME = 'tests.jsonl'

    def get_train_examples(self, data_dir):
        train_file_name = self.TRAIN_FILE_NAME
        train_label_name = self.TRAIN_LABEL_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, train_file_name)),
                                     self._read_jsonl(os.path.join(data_dir, train_label_name)), 'train')

    def get_dev_examples(self, data_dir):
        dev_file_name = self.DEV_FILE_NAME
        dev_label_name = self.DEV_LABEL_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, dev_file_name)),
                                     self._read_jsonl(os.path.join(data_dir, dev_label_name)), 'dev')

    def get_test_examples(self, data_dir):
        test_file_name = self.TEST_FILE_NAME

        return self._create_examples(self._read_jsonl(os.path.join(data_dir, test_file_name)), None, 'test')

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines, labels, set_type):
        examples = []
        if labels is not None:
            for qid, (line, label) in enumerate(zip(lines, labels)):
                context = ""
                question = line["goal"]
                choices = [line["sol1"], line["sol2"]]
                choices = [c + "." if not c.endswith(".") else c for c in choices]
                examples.append(InputExample(
                    qid=qid,
                    question=question,
                    answers=choices,
                    label=label))
        else:
            for qid, line in enumerate(lines):
                context = ""
                question = line["goal"]
                choices = [line["sol1"], line["sol2"]]
                choices = [c + "." if not c.endswith(".") else c for c in choices]
                # label = fields.get('label', None)
                examples.append(InputExample(
                    qid=qid,
                    question=question,
                    answers=choices,
                    label=None))
        return examples


class PIQADataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.proc = PIQAProcessor()

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        else:
            examples = self.proc.get_dev_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.question
        options = ['%s: %s' % (i, option) for i, option in zip('12', example.answers)]
        options = " ".join(options)
        input_ = "context: %s  options: %s </s>" % (input_, options)
        target = "%s </s>" % str(int(example.label) + 1)
        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)


class ANLIProcessor(DataProcessor):
    """Processor for the ANLI data set."""

    LABELS = ['hyp1', 'hyp2']

    TRAIN_FILE_NAME = 'train.jsonl'
    TRAIN_LABEL_NAME = 'train-labels.lst'
    DEV_FILE_NAME = 'dev.jsonl'
    DEV_LABEL_NAME = 'dev-labels.lst'
    TEST_FILE_NAME = 'test.jsonl'

    def get_train_examples(self, data_dir):
        train_file_name = self.TRAIN_FILE_NAME
        train_label_name = self.TRAIN_LABEL_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, train_file_name)),
                                     self._read_jsonl(os.path.join(data_dir, train_label_name)), 'train')

    def get_dev_examples(self, data_dir):
        dev_file_name = self.DEV_FILE_NAME
        dev_label_name = self.DEV_LABEL_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, dev_file_name)),
                                     self._read_jsonl(os.path.join(data_dir, dev_label_name)), 'dev')

    def get_test_examples(self, data_dir):
        test_file_name = self.TEST_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, test_file_name)), None, 'test')

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines, labels, set_type):
        examples = []
        if labels is not None:
            for (line, label) in zip(lines, labels):
                context = ""
                qid = line["story_id"]
                question = line["obs1"] + " " + line["obs2"]
                choices = [line["hyp1"], line["hyp2"]]
                choices = [c + "." if not c.endswith(".") else c for c in choices]
                examples.append(InputExample(
                    qid=qid,
                    question=question,
                    answers=choices,
                    label=label - 1))
        else:
            for line in lines:
                context = ""
                qid = line["story_id"]
                question = line["obs1"] + " " + line["obs2"]
                choices = [line["hyp1"], line["hyp2"]]
                choices = [c + "." if not c.endswith(".") else c for c in choices]
                examples.append(InputExample(
                    qid=qid,
                    question=question,
                    answers=choices,
                    label=None))
        return examples


class ANLIDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.proc = ANLIProcessor()

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        else:
            examples = self.proc.get_dev_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.question
        options = ['%s: %s' % (i, option) for i, option in zip('12', example.answers)]
        options = " ".join(options)
        input_ = "context: %s  options: %s </s>" % (input_, options)
        target = "%s </s>" % str(int(example.label) + 1)

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)


class OBQAProcessor(DataProcessor):
    """Processor for the OpenBook QA (OBQA) data set."""

    LABELS = ['A', 'B', 'C', 'D']

    def __init__(self, use_KB):
        self.use_KB = use_KB
        if self.use_KB:
            self.TRAIN_FILE_NAME = 'train_with_retrieved_facts_datamine.jsonl'
            self.DEV_FILE_NAME = 'dev_with_retrieved_facts_datamine.jsonl'
            self.TEST_FILE_NAME = 'test_with_retrieved_facts_datamine.jsonl'
        else:
            self.TRAIN_FILE_NAME = 'train.jsonl'
            self.DEV_FILE_NAME = 'dev.jsonl'
            self.TEST_FILE_NAME = 'test.jsonl'

    def get_train_examples(self, data_dir):
        train_file_name = self.TRAIN_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, train_file_name)), 'train')

    def get_dev_examples(self, data_dir):
        dev_file_name = self.DEV_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, dev_file_name)), 'dev')

    def get_test_examples(self, data_dir):
        test_file_name = self.TEST_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, test_file_name)), 'test')

    def get_labels(self):
        return [0, 1, 2, 3]

    def _create_examples(self, lines, set_type):
        examples = []
        for line in lines:
            qid = line['id']
            question = line['question']['stem']
            answers = [choice['text'] for choice in sorted(line['question']['choices'], key=lambda c: c['label'])]
            label = self.LABELS.index(line['answerKey'])

            if self.use_KB:
                article = line['question']['retrieved_facts_context']
            else:
                article = None

            examples.append(InputExample(
                qid=qid,
                question=question,
                answers=answers,
                label=label,
                article=article))

        return examples


class OBQADataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512, use_KB=False):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.use_KB = use_KB

        self.inputs = []
        self.targets = []

        self.proc = OBQAProcessor(self.use_KB)

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        else:
            examples = self.proc.get_dev_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.question
        options = ['%s: %s' % (i, option) for i, option in zip('1234', example.answers)]
        options = " ".join(options)

        if not self.use_KB:
            input_ = "context: %s  options: %s </s>" % (input_, options)
        else:
            article = example.article
            input_ = "context: %s  options: %s  article: %s </s>" % (input_, options, article)

        target = "%s </s>" % str(int(example.label) + 1)

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)


# KILT Tasks:
class KILTFEVERProcessor(DataProcessor):
    """Processor for the KILT FEVER data set."""

    LABELS = ['SUPPORTS', 'REFUTES']

    TRAIN_FILE_NAME = 'fever-train-kilt.jsonl'
    DEV_FILE_NAME = 'fever-dev-kilt.jsonl'
    TEST_FILE_NAME = 'fever-test_without_answers-kilt.jsonl'

    def get_train_examples(self, data_dir):
        train_file_name = self.TRAIN_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, train_file_name)), 'train')

    def get_dev_examples(self, data_dir):
        dev_file_name = self.DEV_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, dev_file_name)), 'dev')

    def get_test_examples(self, data_dir):
        test_file_name = self.TEST_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, test_file_name)), 'test')

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines, set_type):
        examples = []
        if set_type != "test":
            for line in lines:
                context = ""
                qid = line["id"]
                question = line["input"]
                choices = self.LABELS
                choices = [c + "." if not c.endswith(".") else c for c in choices]
                label = self.LABELS.index(line["output"][0]["answer"])
                examples.append(InputExample(
                    qid=qid,
                    question=question,
                    answers=choices,
                    label=label))
        else:
            for line in lines:
                context = ""
                qid = line["id"]
                question = line["input"]
                choices = self.LABELS
                choices = [c + "." if not c.endswith(".") else c for c in choices]
                examples.append(InputExample(
                    qid=qid,
                    question=question,
                    answers=choices,
                    label=None))
        return examples


class KILTFEVERDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.proc = KILTFEVERProcessor()

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == "train":
            examples = self.proc.get_train_examples(self.data_dir)
        elif self.type_path == "valid":
            examples = self.proc.get_dev_examples(self.data_dir)
        else:
            examples = self.proc.get_test_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.question
        options = ['%s: %s' % (i, option) for i, option in zip('12', example.answers)]
        options = " ".join(options)
        input_ = "context: %s  options: %s </s>" % (input_, options)
        target = "%s </s>" % str(int(example.label) + 1)

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)


class KILTT2TProcessor(DataProcessor):
    """Processor for the KILT Text to Text data set."""

    def __init__(self, task_type):
        if task_type == "kilt_ay2":
            self.TRAIN_FILE_NAME = 'aidayago2-train-kilt.jsonl'
            self.DEV_FILE_NAME = 'aidayago2-dev-kilt.jsonl'
            self.TEST_FILE_NAME = 'aidayago2-test_without_answers-kilt.jsonl'
        elif task_type == "kilt_natural_qa":
            self.TRAIN_FILE_NAME = 'nq-train-kilt.jsonl'
            self.DEV_FILE_NAME = 'nq-dev-kilt.jsonl'
            self.TEST_FILE_NAME = 'nq-test_without_answers-kilt.jsonl'
        elif task_type == "kilt_trivia_qa":
            self.TRAIN_FILE_NAME = 'triviaqa-train-kilt.jsonl'
            self.DEV_FILE_NAME = 'triviaqa-dev-kilt.jsonl'
            self.TEST_FILE_NAME = 'triviaqa-test_without_answers-kilt.jsonl'
        else:
            raise Exception("Invalid kilt task type: " + task_type)

    def get_train_examples(self, data_dir):
        train_file_name = self.TRAIN_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, train_file_name)), 'train')

    def get_dev_examples(self, data_dir):
        dev_file_name = self.DEV_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, dev_file_name)), 'dev')

    def get_test_examples(self, data_dir):
        test_file_name = self.TEST_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, test_file_name)), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        if set_type != "test":
            for line in lines:
                qid = line["id"]
                input = line["input"]
                output = []
                for cur_out in line["output"]:
                    if cur_out.get("answer") is not None:
                        output.append(cur_out["answer"])

                cur_dict = {
                    "id": qid,
                    "input": input,
                    "output": output
                }
                examples.append(cur_dict)
        else:
            for line in lines:
                qid = line["id"]
                input = line["input"]
                output = None
                cur_dict = {
                    "id": qid,
                    "input": input,
                    "output": output
                }
                examples.append(cur_dict)
        return examples

class KILTT2TDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_source_length=256, max_target_length=32, createMultipleSamples=False):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.createMultipleSamples = createMultipleSamples

        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.task_type = data_dir.split("/")[-1]
        self.proc = KILTT2TProcessor(self.task_type)
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        if self.type_path == "train":
            examples = self.proc.get_train_examples(self.data_dir)
        elif self.type_path == "valid":
            examples = self.proc.get_dev_examples(self.data_dir)
        else:
            examples = self.proc.get_test_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _extractInputForEntityTasks(self, input_string, max_num_tokens=450):
        input_split_list = input_string.split()
        num_tokens = len(input_split_list)
        start_token = "[START_ENT]"
        end_token = "[END_ENT]"
        l_idx = None
        r_idx = None
        for i in range(len(input_split_list)):
            if input_split_list[i] == start_token:
                l_idx = i
            elif input_split_list[i] == end_token:
                r_idx = i

        result = []
        for i in range(l_idx, r_idx + 1, 1):
            result.append(input_split_list[i])

        l_idx -= 1
        r_idx += 1
        break_flag = False
        while not break_flag:
            if l_idx >= 0:
                result = [input_split_list[l_idx]] + result
                l_idx -= 1
            if r_idx <= num_tokens - 1:
                result = result + [input_split_list[r_idx]]
                r_idx += 1

            if l_idx < 0 and r_idx > num_tokens - 1:
                break_flag = True

            if len(result) >= max_num_tokens:
                break_flag = True

        result = " ".join(result)
        return result

    def _normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _create_features(self, example):
        # Create only one instance using the first answer as the only answer to the given input
        if not self.createMultipleSamples:
            if self.task_type == "kilt_natural_qa" or self.task_type == "kilt_trivia_qa":
                input = "question: " + example["input"] + " </s>"
                target_list = [self._normalize_answer(example["output"][0]) + " </s>"]
            elif self.task_type == "kilt_ay2":
                input = "map the entity in the given text: " + self._extractInputForEntityTasks(example["input"])
                target_list = [example["output"][0]]
            else:
                input = example["input"]
                target_list = [example["output"][0]]
        else:
        # Create multiple instances for each correct answer to the given input
            if self.task_type == "kilt_natural_qa" or self.task_type == "kilt_trivia_qa":
                input = "question: " + example["input"]
                target_list = [self._normalize_answer(x) for x in example["output"]]
            elif self.task_type == "kilt_ay2":
                input = "map the entity in the given text: " + self._extractInputForEntityTasks(example["input"])
                target_list = example["output"]
            else:
                input = example["input"]
                target_list = example["output"]

        for target in target_list:
            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_source_length, pad_to_max_length=True, return_tensors="pt", truncation=True
            )

            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_target_length, pad_to_max_length=True, return_tensors="pt", truncation=True
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


class SQUADDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer

        self.inputs = []
        self.targets = []

        self.proc = OBQAProcessor(self.use_KB)

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        else:
            examples = self.proc.get_dev_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.question
        options = ['%s: %s' % (i, option) for i, option in zip('1234', example.answers)]
        options = " ".join(options)

        if not self.use_KB:
            input_ = "context: %s  options: %s </s>" % (input_, options)
        else:
            article = example.article
            input_ = "context: %s  options: %s  article: %s </s>" % (input_, options, article)

        target = "%s </s>" % str(int(example.label) + 1)

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)
