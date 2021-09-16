import argparse
import logging

from trainer import T5FineTuner
from utils import get_best_checkpoint
from torch.utils.data import DataLoader
from transformers import (
    T5Config
)
# Set up logger
logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="t5-base",
        help="Model name or path.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Checkpoint directory.")

    args = parser.parse_args()

    best_checkpoint_path = get_best_checkpoint(args.checkpoint_dir)
    print("Using checkpoint = ", str(best_checkpoint_path))
    classifier = T5FineTuner.load_from_checkpoint(best_checkpoint_path)
    print("SimCSE checkpoint -> Huggingface checkpoint for {}".format(args.checkpoint_dir))
    classifier.model.save_pretrained(args.checkpoint_dir)
    classifier.tokenizer.save_pretrained(args.checkpoint_dir)
    T5Config.from_pretrained(args.model).save_pretrained(args.checkpoint_dir)

if __name__ == "__main__":
    main()
