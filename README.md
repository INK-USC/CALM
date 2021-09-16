# Pre-training Text-to-Text Transformers for Concept-centric Common Sense

This code is for ICLR2021 paper: [Pre-training Text-to-Text Transformers for Concept-centric Common Sense](https://openreview.net/forum?id=3k20LAiHYL2).
Checkout our [Project website](https://inklab.usc.edu/calm-project) for details!

## Installation

```
conda create -n calm python==3.7
conda activate calm
python setup.py install
cd CALM
```

## Preprocess for CALM

### wiki pre-process
```
cat wiki.doc | tail -n +500000 | head -n 500000 > wiki/wiki.train.raw
cat wiki.doc | tail -n +1000000 | head -n 100000 > wiki/wiki.valid.raw
```

### Generative Objective
```
python dataset_utils/concept_deshuffling_data_generation.py
python dataset_utils/keyword_lm_data_generation.py
```

Dataset creation for concept-order-recovering (COR) and concept-to-sentence (C2S).

### Contrastive Objective
```
python dataset_utils/generate_discriminative_dataset.py
```
Dataset creation for generative question answering (QA) dataset.\
There are three types of contrastive objectives (See Table 4 (b) in the [paper](https://openreview.net/forum?id=3k20LAiHYL2)). 
```
Option 1: Multi-choice QA
Option 2: Generative QA
Option 3: True/False
```
For CALM, we use option 2, which is Generative QA.

### Mix three dataset
```
python dataset_utils/mix_dataset.py
```

## Pre-training
### Pre-train CALM_mix
First, train the mix dataset.
```
python finetune.py \
    --data_dir datasets/mix \
    --output_dir outputs/calm_mix_base \
    --model_name_or_path t5-base \
    --tokenizer_name_or_path t5-base \
    --max_seq_length 256 \
    --learning_rate 5e-4 \
    --num_train_epochs 2 \
    --train_batch_size 8 \
    --graident_accumulation_steps 4 \
    --weight_decay 0.01 \
    --warmup_steps 10000 \
    --adam_epsilon 1e-6 \
    --n_gpu 4 \
    --gpu_nums 4,5,6,7 \
    --model_parallel

python finetune.py \
    --data_dir datasets/mix \
    --output_dir outputs/calm_mix_large_dp \
    --model_name_or_path t5-large \
    --tokenizer_name_or_path t5-large \
    --max_seq_length 256 \
    --learning_rate 5e-4 \
    --num_train_epochs 2 \
    --train_batch_size 8 \
    --graident_accumulation_steps 4 \
    --weight_decay 0.01 \
    --warmup_steps 10000 \
    --adam_epsilon 1e-6
```
### Pre-train CALM
Then, train CALM using the checkpoint of mix dataset.
```
python finetune_generator_discriminator.py \
    --data_dir datasets/option2 \
    --checkpoint_dir outputs/calm_mix \
    --output_dir outputs/calm \
    --max_seq_length 256 \
    --learning_rate 5e-7 \
    --num_train_epochs 3 \
    --train_batch_size 8 \
    --graident_accumulation_steps 32 \
    --fp_16 False \
    --weight_decay 0.01 \
    --warmup_steps 10000 \
    --adam_epsilon 1e-6 \
    --n_gpu 8 \
    --gpu_nums 0,1,2,3,4,5,6,7

python finetune_generator_discriminator.py \
    --data_dir datasets/option2 \
    --checkpoint_dir outputs/calm_mix_base_dp \
    --output_dir outputs/calm_base_dp \
    --max_seq_length 256 \
    --learning_rate 5e-7 \
    --num_train_epochs 3 \
    --train_batch_size 8 \
    --graident_accumulation_steps 32 \
    --fp_16 False \
    --weight_decay 0.01 \
    --warmup_steps 10000 \
    --adam_epsilon 1e-6
```

## Fine-tuning

Use checkpoint to fine-tune on the downstream tasks.

## Model List

Our released models are listed as following. You can import these models by using [HuggingFace's Transformers](https://github.com/huggingface/transformers).
 
|              Model              | CSQA | OBQA | PIQA | aNLI | Description
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|
| [danny911kr/calm-mix-base](https://huggingface.co/danny911kr/calm-mix-base) | 63.02 | 60.40 | 70.07 | 62.79 | Mix-Only
| [danny911kr/calm-base](https://huggingface.co/danny911kr/calm-base) | 63.32 | 60.90 | 71.01 | 63.20 | 
| [danny911kr/calm-mix-large](https://huggingface.co/danny911kr/calm-mix-large) | 70.26 | 62.50 | 73.70 | 75.99 | Mix-Only
| [danny911kr/calm-large](https://huggingface.co/danny911kr/calm-large) | 71.31 | 63.90 | 76.90 | 82.35 |

