import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
from PIL import Image, ImageFilter
import math
import random
import numpy as np
import cv2

SOS_TOKEN = 0  # special token for start of sentence
EOS_TOKEN = 1  # special token for end of sentence

class ConvertBetweenStringAndLabel(object):
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

        self.dict = {}
        self.dict['SOS_TOKEN'] = SOS_TOKEN
        self.dict['EOS_TOKEN'] = EOS_TOKEN
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i + 2

    def encode(self, text):
        """
        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor targets:max_length × batch_size
        """
        if isinstance(text, str):
            text = [self.dict[item] if item in self.dict else 2 for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]
            max_length = max([len(x) for x in text])
            nb = len(text)
            targets = torch.ones(nb, max_length + 2) * 2
            for i in range(nb):
                targets[i][0] = 0
                targets[i][1:len(text[i]) + 1] = text[i]
                targets[i][len(text[i]) + 1] = 1
            text = targets.transpose(0, 1).contiguous()
            text = text.long()
        return torch.LongTensor(text)

    def decode(self, t):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """

        texts = list(self.dict.keys())[list(self.dict.values()).index(t)]
        return texts

class Averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def load_data(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def weights_init(model):
    # Official init from torch repo.
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)
