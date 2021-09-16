from tqdm import tqdm
import os
import argparse
from generator.concept.concept_generator import *

generator = ConceptGenerator()

def match_sents(sent_path):
    result = []
    with open(sent_path) as f:
        sents = f.read().split("\n")

    for index, sent in enumerate(tqdm(sents)):
        if sent == "":
            continue

        if generator.check_availability(sent):
            generated_sentence = generator.cor_generate(sent)
            result.append({"original_sentence": sent, "output_sentence": generated_sentence})
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='datasets/wiki')
parser.add_argument('--save_path', default="datasets/cor")
parser.add_argument('--num_batch', default=30, type=int)
parser.add_argument('--batch_id', default=0, type=int)
args = parser.parse_args()

batch_result = match_sents(sent_path=os.path.join(args.input_path, "wiki.train.raw"))

with open(os.path.join(args.save_path, "train.source"), "w") as f:
    for line in batch_result:
        f.write("correct the order of the following sentence: "+line["output_sentence"]+'\n')

with open(os.path.join(args.save_path, "train.target"), "w") as f:
    for line in batch_result:
        f.write(line["original_sentence"]+'\n')

batch_result = match_sents(sent_path=os.path.join(args.input_path, "wiki.valid.raw"))

with open(os.path.join(args.save_path, "dev.source"), "w") as f:
    for line in batch_result:
        f.write("correct the order of the following sentence: "+line["output_sentence"]+'\n')

with open(os.path.join(args.save_path, "dev.target"), "w") as f:
    for line in batch_result:
        f.write(line["original_sentence"]+'\n')
