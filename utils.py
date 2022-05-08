import os
import random
import numpy as np
import pandas as pd
import logging

import torch
from torch.backends.cudnn import deterministic
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_test_examples(filename):
    """Read examples from filename."""
    sum_examples = []
    gen_examples = []
    df = pd.read_csv(filename)
    input_text = df['input_text'].tolist()
    target_text = df['target_text'].tolist()
    for i in range(len(input_text)):
        if("ShellCodeSum" in input_text[i]):
            sum_examples.append(
                Example(
                    idx=i,
                    source=str(input_text[i]),
                    target=str(target_text[i]),
                )
            )
        else:
            gen_examples.append(
                Example(
                    idx=i,
                    source=str(input_text[i]),
                    target=str(target_text[i]),
                )
            )
    return sum_examples, gen_examples

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    df = pd.read_csv(filename)
    input_text = df['input_text'].tolist()
    target_text = df['target_text'].tolist()
    for i in range(len(input_text)):
        examples.append(
            Example(
                idx=i,
                source=str(input_text[i]),
                target=str(target_text[i]),
            )
        )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples, desc='convert examples to features...')):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if example_index < 3:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
