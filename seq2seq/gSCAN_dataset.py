import os
from typing import List
import logging
from collections import defaultdict
from collections import Counter
import json
import torch

from GroundedScan.dataset import GroundedScan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class Vocabulary(object):

    def __init__(self, unk_token="<UNK>", sos_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>"):
        """
        NB: that unknown words will map to <UNK>. <PAD> token is by construction idx 0.
        """
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self._idx_to_word = [pad_token, unk_token, sos_token, eos_token]
        self._word_to_idx = defaultdict(lambda: self._idx_to_word.index(self.unk_token))
        self._word_frequencies = Counter()

    def word_to_idx(self, word: str) -> int:
        return self._word_to_idx[word]

    def idx_to_word(self, idx: int) -> str:
        return self._idx_to_word[idx]

    def add_sentence(self, sentence: List[str]):
        for word in sentence:
            if word not in self._word_to_idx:
                self._word_to_idx[word] = self.size
                self._idx_to_word.append(word)
            self._word_frequencies[word] += 1

    def most_common(self, n=10):
        return self._word_frequencies.most_common(n=n)

    @property
    def pad_idx(self):
        return self.word_to_idx(self.pad_token)

    @property
    def size(self):
        return len(self._idx_to_word)

    @classmethod
    def load(cls, path: str):
        assert os.path.exists(path), "Trying to load a vocabulary from a non-existing file {}".format(path)
        with open(path, 'r') as infile:
            all_data = json.load(infile)
            unk_token = all_data["unk_token"]
            sos_token = all_data["sos_token"]
            eos_token = all_data["eos_token"]
            pad_token = all_data["pad_token"]
            vocab = cls(unk_token=unk_token, sos_token=sos_token, eos_token=eos_token, pad_token=pad_token)
            vocab._idx_to_word = all_data["idx_to_word"]
            vocab._word_to_idx = defaultdict(int)
            for word, idx in all_data["word_to_idx"].items():
                vocab._word_to_idx[word] = idx
            vocab._word_frequencies = Counter(all_data["word_frequencies"])
        return vocab

    def to_dict(self) -> dict:
        return {
            "unk_token": self.unk_token,
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "idx_to_word": self._idx_to_word,
            "word_to_idx": self._word_to_idx,
            "word_frequencies": self._word_frequencies
        }

    def save(self, path: str) -> str:
        with open(path, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)
        return path


class GroundedScanDataset(object):
    """

    """

    def __init__(self, path_to_data: str, save_directory: str, split="train", input_vocabulary_file="",
                 target_vocabulary_file="", generate_vocabulary=False):
        assert os.path.exists(path_to_data), "Trying to read a gSCAN dataset from a non-existing file {}.".format(
            path_to_data)
        if not generate_vocabulary:
            assert os.path.exists(input_vocabulary_file) and os.path.exists(target_vocabulary_file), \
                "Trying to load vocabularies from non-existing files."
        if split == "test" and generate_vocabulary:
            logger.warning("WARNING: generating a vocabulary from the test set.")
        self.dataset = GroundedScan.load_dataset_from_file(path_to_data, save_directory=save_directory)
        self.split = split
        if generate_vocabulary:
            logger.info("Generating vocabularies...")
            self.input_vocabulary = Vocabulary()
            self.target_vocabulary = Vocabulary()
            self.read_vocabularies()
        else:
            logger.info("Loading vocabularies...")
            self.input_vocabulary = Vocabulary.load(input_vocabulary_file)
            self.target_vocabulary = Vocabulary.load(target_vocabulary_file)

    def read_vocabularies(self):
        logger.info("Populating vocabulary...")
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

    def get_data_batch(self, batch_size=10):
        # TODO: think more about this and efficiency
        batch = []
        lengths = []
        max_length = 0
        for example in self.dataset.get_examples_with_image(self.split):
            input_commands = example["input_command"]
            if len(batch) == batch_size:
                break
            batch.append(self.sentence_to_array(input_commands, vocabulary="input"))
            lengths.append(len(input_commands))
            if len(input_commands) > max_length:
                max_length = len(input_commands)
        # Pad the batch with zero's
        for example in batch:
            num_to_pad = max_length - len(example)
            example.extend([self.input_vocabulary.pad_idx] * num_to_pad)
        return torch.tensor(batch, dtype=torch.long, device=device), lengths

    def sentence_to_array(self, sentence: List[str], vocabulary: str):
        vocab = self.get_vocabulary(vocabulary)
        return [vocab.word_to_idx(word) for word in sentence]

    def array_to_sentence(self, sentence_array: List[int], vocabulary: str):
        vocab = self.get_vocabulary(vocabulary)
        return [vocab.idx_to_word(word_idx) for word_idx in sentence_array]

    @property
    def input_vocabulary_size(self):
        return self.input_vocabulary.size

    @property
    def target_vocabulary_size(self):
        return self.target_vocabulary.size
