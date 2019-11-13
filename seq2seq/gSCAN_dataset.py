import os
from typing import List
import logging
from collections import defaultdict
import json

from GroundedScan.dataset import GroundedScan


class Vocabulary(object):

    def __init__(self, unk_token="<UNK>"):
        """
        NB: that unknown words will map to <UNK> with idx 0.
        """
        self.unk_token = unk_token
        self.idx_to_word = [unk_token]
        self.word_to_idx = defaultdict(int)

    def add_sentence(self, sentence: List[str]):
        for word in sentence:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.size
                self.idx_to_word.append(word)

    @property
    def size(self):
        return len(self.idx_to_word)

    @classmethod
    def load(cls, path: str, unk_token="<UNK>"):
        assert os.path.exists(path), "Trying to load a vocabulary from a non-existing file {}".format(path)
        with open(path, 'r') as infile:
            all_data = json.load(infile)
            vocab = cls(unk_token=unk_token)
            vocab.unk_token = all_data["unk_token"]
            vocab.idx_to_word = all_data["idx_to_word"]
            vocab.word_to_idx = all_data["word_to_idx"]
        return vocab

    def to_dict(self) -> dict:
        return {
            "unk_token": self.unk_token,
            "idx_to_word": self.idx_to_word,
            "word_to_idx": self.word_to_idx
        }

    def save(self, path: str) -> str:
        with open(path, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)
        return path


class GroundedScanDataset(object):
    """

    """

    def __init__(self, path_to_data: str, save_directory: str, split="train", input_vocabulary_file="",
                 target_vocabulary_file=""):
        assert os.path.exists(path_to_data), "Trying to read a gSCAN dataset from a non-existing file {}.".format(
            path_to_data)
        self.dataset = GroundedScan.load_dataset_from_file(path_to_data, save_directory=save_directory)
        self.split = split
        if not input_vocabulary_file or not target_vocabulary_file:
            self.input_vocabulary = Vocabulary()
            self.target_vocabulary = Vocabulary()
            self.read_vocabularies()
        else:
            print("Loading vocabularies...")
            self.input_vocabulary = Vocabulary.load(input_vocabulary_file)
            self.target_vocabulary = Vocabulary.load(target_vocabulary_file)

    def read_vocabularies(self):
        print("Populating vocabulary...")
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split)):
            self.input_vocabulary.add_sentence(example["input_command"])
            self.target_vocabulary.add_sentence(example["target_command"])

    def save_vocabularies(self, input_vocabulary_file: str, target_vocabulary_file: str):
        self.input_vocabulary.save(input_vocabulary_file)
        self.target_vocabulary.save(target_vocabulary_file)

    def get_vocabulary(self, vocabulary: str) -> Vocabulary:
        if vocabulary == "input":
            vocab = self.input_vocabulary
        elif vocabulary == "target":
            vocab = self.target_vocabulary
        else:
            raise ValueError("Specified unknown vocabulary in sentence_to_array: {}".format(vocabulary))
        return vocab

    def sentence_to_array(self, sentence: List[str], vocabulary: str):
        vocab = self.get_vocabulary(vocabulary)
        return [vocab.word_to_idx[word] for word in sentence]

    def array_to_sentence(self, sentence_array: List[int], vocabulary: str):
        vocab = self.get_vocabulary(vocabulary)
        return [vocab.idx_to_word[word_idx] for word_idx in sentence_array]

    @property
    def input_vocabulary_size(self):
        return self.input_vocabulary.size

    @property
    def target_vocabulary_size(self):
        return self.target_vocabulary.size
