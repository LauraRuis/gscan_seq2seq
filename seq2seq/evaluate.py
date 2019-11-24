from seq2seq.predict import predict
from seq2seq.helpers import sequence_accuracy

import torch.nn as nn
from typing import Iterator
import numpy as np


def evaluate(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, sos_idx: int, eos_idx: int) -> float:
    accuracies = []
    exact_match = 0
    for input_sequence, _, output_sequence, target_sequence, _, _ in predict(data_iterator=data_iterator, model=model,
                                                                             max_decoding_steps=max_decoding_steps,
                                                                             sos_idx=sos_idx, eos_idx=eos_idx):
        accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
        if accuracy == 100:
            exact_match += 1
        accuracies.append(accuracy)
    return np.mean(np.array(accuracies)), (exact_match / len(accuracies)) * 100
