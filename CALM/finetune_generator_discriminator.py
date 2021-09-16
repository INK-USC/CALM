"""Fine-tune T5-based model."""

import argparse
import os

import pytorch_lightning as pl
import torch

from logger import LoggingCallback
from trainer_generator_discriminator import T5GANFineTuner
from utils import get_best_checkpoint, set_seed


def main() -> None:
    """Fine-tune T5-based model."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data_dir", type=str, default="task_data/semeval-t5", help="Path for data files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/t5-large",
        help="Path to save the checkpoints.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Checkpoint directory.")
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=-1,
        help="Interval of training steps to save the model checkpoints."
        "Use -1 to disable this callback.",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="t5-large",
        help="Model name or path.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="t5-large",
        help="Tokenizer name or path.",
    )

    # You can find out more on optimization levels at
    # https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    parser.add_argument("--opt_level", type=str, default="O1", help="Optimization level.")
    parser.add_argument(
        "--early_stop_callback", action="store_true", help="Whether to do early stopping."
    )

    # If you want to enable 16-bit training, set this to true
    parser.add_argument(
        "--fp_16",
        action="store_true",
        help="Whether to use 16-bit precision floating point operations.",
    )

    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-8, help="Epsilon value for Adam optimizer."
    )

    # If you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm value for clipping."
    )

    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--warmup_steps", type=int, default=400, help="Number of warmup steps.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--n_gpu", type=int, default=1, help="Number of GPUs to use for computation."
    )
    parser.add_argument(
        "--gpu_nums",
        type=str,
        default="-1",
        help='GPU IDs separated by "," to use for computation.',
    )
    parser.add_argument("--seed", type=int, default=42, help="Manual seed value.")
    parser.add_argument("--model_parallel", action="store_true", help="Enable model parallelism.")

    args = parser.parse_known_args()[0]
    print(args)

    set_seed(args.seed)

    # Create a folder if output_dir doesn't exist
    if not os.path.exists(args.output_dir):
        print("Creating output directory")
        os.makedirs(args.output_dir)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir + "/{epoch}-{val_loss:.6f}",
        prefix="checkpoint_",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    if args.checkpoint_dir:
        best_checkpoint_path = get_best_checkpoint(args.checkpoint_dir)
        print(f"Using checkpoint = {best_checkpoint_path}")
        checkpoint_state = torch.load(best_checkpoint_path, map_location="cpu")
        model = T5GANFineTuner(args)
        model.load_state_dict(checkpoint_state["state_dict"])
    else:
        model = T5GANFineTuner(args)

    trainer = pl.Trainer(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        early_stop_callback=args.early_stop_callback,
        max_epochs=args.num_train_epochs,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        distributed_backend="ddp" if not args.model_parallel else None,
        gpus=args.gpu_nums if not args.model_parallel else None,
        callbacks=[LoggingCallback()],
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
