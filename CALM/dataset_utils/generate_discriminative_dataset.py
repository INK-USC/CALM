# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import uuid
import tensorflow.compat.v1 as tf
from torch.utils.data import Dataset
import os, glob
from generator.concept.concept_generator import *
from tqdm import tqdm
import tensorflow_datasets as tfds
from transformers import (
    T5Tokenizer
)

class Option1Dataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512, option=1):
        self.type_path = type_path
        self.file_path = os.path.join(data_dir)
        self.files = glob.glob("%s/wiki.%s.raw" % (self.file_path, type_path))

        self.option = option #option = 1 (number) =2 (text)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.generator = ConceptGenerator()

        self.source_text = []
        self.target_text = []

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
        self._build_examples_from_files(self.files)

        source = open('datasets/option1/' + self.type_path+'.source', 'w')
        target = open('datasets/option1/' + self.type_path+'.target', 'w')

        for st, tt in zip(self.source_text, self.target_text):
            source.write("%s\n" % st)
            target.write("%s\n" % tt)

        source.close()
        target.close()


    def neighboring_pairs_test(self, dataset, text_key='text'):
        def split_by_lines(dataset):
            """Splits text in dataset by line, removing empty lines."""
            def my_fn(text):
                lines = tf.strings.split([text], sep='\n').values
                return tf.strings.strip(lines)
            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def split_by_sep(dataset):
            """Splits text in dataset by line, removing empty lines."""
            def my_fn(text):
                sep = str(uuid.uuid4())
                sentences = tf.strings.regex_replace(text, r'((?:\.|\!|\?)+)', r'\1' + sep)
                sentences = tf.strings.strip(tf.strings.split([sentences], sep).values)
                return sentences
            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def get_sentence(line):
            return {
                'text': line,
            }

        # Split by lines.
        dataset = dataset.map(lambda x: x[text_key], num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = split_by_lines(dataset)
        dataset = split_by_sep(dataset)
        dataset = dataset.map(get_sentence, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def filter_fn(x):
            return self.generator.check_availability(x)

        dataset = dataset.filter(filter_fn)
        return dataset

    def neighboring_pairs_train(self, dataset, text_key='text'):
        def split_by_lines(dataset):
            """Splits text in dataset by line, removing empty lines."""
            def my_fn(text):
                lines = tf.strings.split([text], sep='\n\n').values
                return tf.strings.strip(lines)
            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def split_by_sep(dataset):
            """Splits text in dataset by line, removing empty lines."""
            def my_fn(text):
                sentences = tf.strings.strip(tf.strings.split([text], sep='\n').values)
                return sentences
            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def get_sentence(line):
            return {
                'text': line,
            }

        # Split by lines.
        dataset = dataset.map(lambda x: x[text_key], num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = split_by_lines(dataset)
        dataset = split_by_sep(dataset)
        dataset = dataset.map(get_sentence, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def filter_fn(x):
            return self.generator.check_availability(x)

        dataset = dataset.filter(filter_fn)
        return dataset

    def _build_examples_from_files(self, files, label='Which sentence is correct ?: '):
        for path in files:
            with open(path, 'r') as f:
                text = f.read()

            og_dataset = tf.data.Dataset.from_tensor_slices({'text': [text]})
            empty = tf.constant('', dtype=tf.string, shape=[1])
            if self.type_path == 'train':
                dataset = self.neighboring_pairs_train(og_dataset, text_key='text')
            else:
                dataset = self.neighboring_pairs_test(og_dataset, text_key='text')
            dataset = dataset.shuffle(500000)
            dataset_length = [i for i, _ in enumerate(tfds.as_numpy(dataset))][-1] + 1
            print(dataset_length)

            def some_are_empty(*tensors):
                """See if at least one tensor has shape [0]."""
                empty = [tf.equal(tf.size(t), 0) for t in tensors]
                return tf.reduce_any(empty)

            def my_fn(x):
                """Function to be applied to each example in dataset."""
                negative_sampling = tf.random.uniform(shape=[]) < 0.5

                def get_generated_sentence(sentence):
                    # you should decode bytes type to string type
                    generated_sentences = []
                    generated_sentence = self.generator.cor_generate(sentence.numpy().decode('utf-8'))
                    generated_sentences.append(tf.convert_to_tensor(generated_sentence, dtype=tf.string))
                    return tf.stack(generated_sentences)

                #TODO: add reconstructor
                # recover_sentence = recover(generated-sentence)

                encode_sentence = tf.py_function(get_generated_sentence, [x['text']], [tf.string])[0]
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    with sess.as_default():
                        encode_sentence.set_shape(x['text'].get_shape())


                concept_option_1, concept_option_2 = tf.cond(
                    negative_sampling,
                    lambda: (x['text'], encode_sentence), # 0 : positive
                    lambda: (encode_sentence, x['text']), # 1 : positive
                )

                target_label = tf.cond(
                    negative_sampling,
                    lambda: "1 </s>",
                    lambda: "2 </s>",
                )

                inputs = []

                def create_examples(first_i=concept_option_1, second_i=concept_option_2):
                    return tf.strings.join([
                        label, # 'Which sentence is correct ?: '
                        'options: ', #options:
                        '1: ', #1:
                        first_i,
                        ' ',
                        '2: ', #2:
                        second_i,
                        ' ',
                        '</s>',
                    ])

                inpt = tf.cond(
                    some_are_empty(concept_option_1, concept_option_2),
                    lambda: empty,
                    create_examples,
                )

                inputs.append(tf.strings.strip(inpt))
                inputs = tf.reshape(inputs, [-1])
                targets = tf.reshape(1 * [target_label], [-1])
                return {'inputs': inputs, 'targets': targets}

            dataset = dataset.map(my_fn)
            dataset = dataset.unbatch()

            def example_len(x):
                return tf.math.minimum(
                    tf.strings.length(x['inputs']), tf.strings.length(x['targets']))

            dataset = dataset.filter(lambda x: example_len(x) > 0)

            for i, data in tqdm(enumerate(tfds.as_numpy(dataset))):
                if len(data['inputs'].decode('utf-8').split()) > self.max_len:
                    continue
                self.source_text.append(data['inputs'].decode('utf-8'))
                self.target_text.append(data['targets'].decode('utf-8'))


class Option2Dataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512, option=1):
        self.type_path = type_path
        self.file_path = os.path.join(data_dir)
        self.files = glob.glob("%s/wiki.%s.raw" % (self.file_path, type_path))

        self.option = option #option = 1 (number) =2 (text)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.generator = ConceptGenerator()

        self.source_text = []
        self.target_text = []

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
        self._build_examples_from_files(self.files)

        source = open('datasets/option2/' + self.type_path+'.source', 'w')
        target = open('datasets/option2/' + self.type_path+'.target', 'w')

        for st, tt in zip(self.source_text, self.target_text):
            source.write("%s\n" % st)
            target.write("%s\n" % tt)

        source.close()
        target.close()


    def neighboring_pairs_test(self, dataset, text_key='text'):
        def split_by_lines(dataset):
            """Splits text in dataset by line, removing empty lines."""
            def my_fn(text):
                lines = tf.strings.split([text], sep='\n').values
                return tf.strings.strip(lines)
            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def split_by_sep(dataset):
            """Splits text in dataset by line, removing empty lines."""
            def my_fn(text):
                sep = str(uuid.uuid4())
                sentences = tf.strings.regex_replace(text, r'((?:\.|\!|\?)+)', r'\1' + sep)
                sentences = tf.strings.strip(tf.strings.split([sentences], sep).values)
                return sentences
            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def get_sentence(line):
            return {
                'text': line,
            }

        # Split by lines.
        dataset = dataset.map(lambda x: x[text_key], num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = split_by_lines(dataset)
        dataset = split_by_sep(dataset)
        dataset = dataset.map(get_sentence, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def filter_fn(x):
            return self.generator.check_availability(x)

        dataset = dataset.filter(filter_fn)
        return dataset

    def neighboring_pairs_train(self, dataset, text_key='text'):
        def split_by_lines(dataset):
            """Splits text in dataset by line, removing empty lines."""
            def my_fn(text):
                lines = tf.strings.split([text], sep='\n\n').values
                return tf.strings.strip(lines)
            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def split_by_sep(dataset):
            """Splits text in dataset by line, removing empty lines."""
            def my_fn(text):
                sentences = tf.strings.strip(tf.strings.split([text], sep='\n').values)
                return sentences
            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def get_sentence(line):
            return {
                'text': line,
            }

        # Split by lines.
        dataset = dataset.map(lambda x: x[text_key], num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = split_by_lines(dataset)
        dataset = split_by_sep(dataset)
        dataset = dataset.map(get_sentence, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def filter_fn(x):
            return self.generator.check_availability(x)

        dataset = dataset.filter(filter_fn)
        return dataset

    def _build_examples_from_files(self, files, label='Which sentence is correct ?: '):
        for path in files:
            with open(path, 'r') as f:
                text = f.read()

            og_dataset = tf.data.Dataset.from_tensor_slices({'text': [text]})
            empty = tf.constant('', dtype=tf.string, shape=[1])
            if self.type_path == 'train':
                dataset = self.neighboring_pairs_train(og_dataset, text_key='text')
            else:
                dataset = self.neighboring_pairs_test(og_dataset, text_key='text')
            dataset = dataset.shuffle(500000)
            dataset_length = [i for i, _ in enumerate(tfds.as_numpy(dataset))][-1] + 1
            print(dataset_length)

            def some_are_empty(*tensors):
                """See if at least one tensor has shape [0]."""
                empty = [tf.equal(tf.size(t), 0) for t in tensors]
                return tf.reduce_any(empty)

            def my_fn(x):
                """Function to be applied to each example in dataset."""
                negative_sampling = tf.random.uniform(shape=[]) < 0.5

                def get_generated_sentence(sentence):
                    # you should decode bytes type to string type
                    generated_sentences = []
                    generated_sentence = self.generator.cor_generate(sentence.numpy().decode('utf-8'))
                    generated_sentences.append(tf.convert_to_tensor(generated_sentence, dtype=tf.string))
                    return tf.stack(generated_sentences)

                #TODO: add reconstructor
                # recover_sentence = recover(generated-sentence)

                encode_sentence = tf.py_function(get_generated_sentence, [x['text']], [tf.string])[0]
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    with sess.as_default():
                        encode_sentence.set_shape(x['text'].get_shape())


                concept_option_1, concept_option_2 = tf.cond(
                    negative_sampling,
                    lambda: (x['text'], encode_sentence), # 0 : positive
                    lambda: (encode_sentence, x['text']), # 1 : positive
                )

                target_label = tf.cond(
                    negative_sampling,
                    lambda: x['text'] + " </s>",
                    lambda: x['text'] + " </s>",
                )

                inputs = []

                def create_examples(first_i=concept_option_1, second_i=concept_option_2):
                    return tf.strings.join([
                        label, # 'Which sentence is correct ?: '
                        'options: ', #options:
                        '1: ', #1:
                        first_i,
                        ' ',
                        '2: ', #2:
                        second_i,
                        ' ',
                        '</s>',
                    ])

                inpt = tf.cond(
                    some_are_empty(concept_option_1, concept_option_2),
                    lambda: empty,
                    create_examples,
                )

                inputs.append(tf.strings.strip(inpt))
                inputs = tf.reshape(inputs, [-1])
                targets = tf.reshape(1 * [target_label], [-1])
                return {'inputs': inputs, 'targets': targets}

            dataset = dataset.map(my_fn)
            dataset = dataset.unbatch()

            def example_len(x):
                return tf.math.minimum(
                    tf.strings.length(x['inputs']), tf.strings.length(x['targets']))

            dataset = dataset.filter(lambda x: example_len(x) > 0)

            for i, data in tqdm(enumerate(tfds.as_numpy(dataset))):
                if len(data['inputs'].decode('utf-8').split()) > self.max_len:
                    continue
                self.source_text.append(data['inputs'].decode('utf-8'))
                self.target_text.append(data['targets'].decode('utf-8'))


class Option3Dataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512, option=1):
        self.type_path = type_path
        self.file_path = os.path.join(data_dir)
        self.files = glob.glob("%s/wiki.%s.raw" % (self.file_path, type_path))

        self.option = option #option = 1 (number) =2 (text)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.generator = ConceptGenerator()

        self.source_text = []
        self.target_text = []

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
        self._build_examples_from_files(self.files)

        source = open('datasets/option3/' + self.type_path+'.source', 'w')
        target = open('datasets/option3/' + self.type_path+'.target', 'w')

        for st, tt in zip(self.source_text, self.target_text):
            source.write("%s\n" % st)
            target.write("%s\n" % tt)

        source.close()
        target.close()

    def neighboring_pairs_test(self, dataset, text_key='text'):
        def split_by_lines(dataset):
            """Splits text in dataset by line, removing empty lines."""

            def my_fn(text):
                lines = tf.strings.split([text], sep='\n').values
                return tf.strings.strip(lines)

            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def split_by_sep(dataset):
            """Splits text in dataset by line, removing empty lines."""

            def my_fn(text):
                sep = str(uuid.uuid4())
                sentences = tf.strings.regex_replace(text, r'((?:\.|\!|\?)+)', r'\1' + sep)
                sentences = tf.strings.strip(tf.strings.split([sentences], sep).values)
                return sentences

            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def get_sentence(line):
            return {
                'text': line,
            }

        # Split by lines.
        dataset = dataset.map(lambda x: x[text_key], num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = split_by_lines(dataset)
        dataset = split_by_sep(dataset)
        dataset = dataset.map(get_sentence, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def filter_fn(x):
            return self.generator.check_availability(x)

        dataset = dataset.filter(filter_fn)
        return dataset

    def neighboring_pairs_train(self, dataset, text_key='text'):
        def split_by_lines(dataset):
            """Splits text in dataset by line, removing empty lines."""

            def my_fn(text):
                lines = tf.strings.split([text], sep='\n\n').values
                return tf.strings.strip(lines)

            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def split_by_sep(dataset):
            """Splits text in dataset by line, removing empty lines."""

            def my_fn(text):
                sentences = tf.strings.strip(tf.strings.split([text], sep='\n').values)
                return sentences

            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def get_sentence(line):
            return {
                'text': line,
            }

        # Split by lines.
        dataset = dataset.map(lambda x: x[text_key], num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = split_by_lines(dataset)
        dataset = split_by_sep(dataset)
        dataset = dataset.map(get_sentence, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def filter_fn(x):
            return self.generator.check_availability(x)

        dataset = dataset.filter(filter_fn)
        return dataset

    def _build_examples_from_files(self, files, label='Does this sentence make sense?: '):
        for path in files:
            with open(path, 'r') as f:
                text = f.read()

            og_dataset = tf.data.Dataset.from_tensor_slices({'text': [text]})
            empty = tf.constant('', dtype=tf.string, shape=[1])
            if self.type_path == 'train':
                dataset = self.neighboring_pairs_train(og_dataset, text_key='text')
            else:
                dataset = self.neighboring_pairs_test(og_dataset, text_key='text')
            dataset = dataset.shuffle(500000)
            dataset_length = [i for i, _ in enumerate(tfds.as_numpy(dataset))][-1] + 1
            print(dataset_length)

            def some_are_empty(*tensors):
                """See if at least one tensor has shape [0]."""
                empty = [tf.equal(tf.size(t), 0) for t in tensors]
                return tf.reduce_any(empty)

            def my_fn(x):
                """Function to be applied to each example in dataset."""
                negative_sampling = tf.random.uniform(shape=[]) < 0.5

                def get_generated_sentence(sentence):
                    # you should decode bytes type to string type
                    generated_sentences = []
                    generated_sentence = self.generator.cor_generate(sentence.numpy().decode('utf-8'))
                    generated_sentences.append(tf.convert_to_tensor(generated_sentence, dtype=tf.string))
                    return tf.stack(generated_sentences)

                encode_sentence = tf.py_function(get_generated_sentence, [x['text']], [tf.string])[0]
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    with sess.as_default():
                        encode_sentence.set_shape(x['text'].get_shape())

                concept_sentence = tf.cond(
                    negative_sampling,
                    lambda: (x['text']),
                    lambda: (encode_sentence),
                )

                relation_label = tf.cond(
                    negative_sampling,
                    lambda: 'true </s>',
                    lambda: 'false </s>',
                )

                inputs = []
                concept_input = concept_sentence

                def create_examples(concept_i=concept_input):
                    return tf.strings.join([
                        label,
                        concept_i
                    ])

                inpt = tf.cond(
                    some_are_empty(concept_input),
                    lambda: empty,
                    create_examples,
                )

                inputs.append(tf.strings.strip(inpt))

                inputs = tf.reshape(inputs, [-1])
                targets = tf.reshape(1 * [relation_label], [-1])
                return {'inputs': inputs, 'targets': targets}

            dataset = dataset.map(my_fn)
            dataset = dataset.unbatch()

            def example_len(x):
                return tf.math.minimum(
                    tf.strings.length(x['inputs']), tf.strings.length(x['targets']))

            dataset = dataset.filter(lambda x: example_len(x) > 0)

            for i, data in tqdm(enumerate(tfds.as_numpy(dataset))):
                if len(data['inputs'].decode('utf-8').split()) > self.max_len:
                    continue
                self.source_text.append(data['inputs'].decode('utf-8'))
                self.target_text.append(data['targets'].decode('utf-8'))

tokenizer = T5Tokenizer.from_pretrained("t5-base")
Option2Dataset(tokenizer=tokenizer, data_dir="datasets/wiki", type_path="train", max_len=256)
Option2Dataset(tokenizer=tokenizer, data_dir="datasets/wiki", type_path="dev", max_len=256)

Option1Dataset(tokenizer=tokenizer, data_dir="datasets/wiki", type_path="train", max_len=256)
Option1Dataset(tokenizer=tokenizer, data_dir="datasets/wiki", type_path="dev", max_len=256)

Option3Dataset(tokenizer=tokenizer, data_dir="datasets/wiki", type_path="train", max_len=256)
Option3Dataset(tokenizer=tokenizer, data_dir="datasets/wiki", type_path="dev", max_len=256)
