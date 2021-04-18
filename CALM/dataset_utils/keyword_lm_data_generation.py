import spacy
import json
import argparse
from tqdm import tqdm
import numpy as np
import copy
import random
import os

nlp = spacy.load('en_core_web_sm')

def match_sents(sent_path, num_batch, batch_id):
    #assert sent_path.endswith(".txt")
    with open(sent_path) as f:
        sents = f.read().split("\n")
    result = []
    batches = np.array_split(sents, num_batch)
    batch_sents = list(batches[batch_id])
    for index, sent in enumerate(tqdm(batch_sents, desc="Batch ID: %d" % batch_id)):
        if sent == "":
            continue
        doc = nlp(str(sent))
        matched_concepts = []
        for token in doc:
            if (token.pos_.startswith('V') or token.pos_.startswith('PROP')) and token.is_alpha and not token.is_stop:
                matched_concepts.append(token.lemma_)
        for noun_chunk in doc.noun_chunks:
            root_noun = noun_chunk[-1]
            if root_noun.pos_ == "NOUN":
                matched_concepts.append(root_noun.lemma_)

        result.append({"sentence": sent, "matched_concepts": matched_concepts})
    return result


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='datasets/wiki')
parser.add_argument('--save_path', default="datasets/c2s")
parser.add_argument('--num_batch', default=30, type=int)
parser.add_argument('--batch_id', default=0, type=int)
args = parser.parse_args()

batch_result = match_sents(sent_path=os.path.join(args.input_path, "wiki.train.raw"), num_batch=args.num_batch, batch_id=args.batch_id)

with open(os.path.join(args.save_path, "train.source"), "w") as f:
    for line in batch_result:
        f.write("generate a sentence with the following concepts: "+' '.join(line["matched_concepts"])+'\n')

with open(os.path.join(args.save_path, "train.target"), "w") as f:
    for line in batch_result:
        f.write(line["sentence"]+'\n')

batch_result = match_sents(sent_path=os.path.join(args.input_path, "wiki.valid.raw"), num_batch=args.num_batch, batch_id=args.batch_id)

with open(os.path.join(args.save_path, "valid.source"), "w") as f:
    for line in batch_result:
        f.write("generate a sentence with the following concepts: "+' '.join(line["matched_concepts"])+'\n')

with open(os.path.join(args.save_path, "valid.target"), "w") as f:
    for line in batch_result:
        f.write(line["sentence"]+'\n')
