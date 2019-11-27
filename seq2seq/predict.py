import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Iterator
import time
import json

from seq2seq.helpers import sequence_accuracy
from seq2seq.gSCAN_dataset import GroundedScanDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def predict_and_save(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str, max_decoding_steps: int,
                     max_testing_examples=None,
                     **kwargs):
    """
    TODO
    :param dataset:
    :param model:
    :param output_file_path:
    :param max_decoding_steps:
    :param max_testing_examples:
    :param kwargs:
    :return:
    """
    cfg = locals().copy()

    dataset.read_dataset(max_examples=max_testing_examples)
    logger.info("Done Loading data.")

    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            for (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                 attention_weights_commands, attention_weights_situations) in predict(
                    dataset.get_data_iterator(batch_size=1), model=model, max_decoding_steps=max_decoding_steps,
                    pad_idx=dataset.target_vocabulary.pad_idx, sos_idx=dataset.target_vocabulary.sos_idx,
                    eos_idx=dataset.target_vocabulary.eos_idx):
                accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
                input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
                target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
                output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                               "derivation": derivation_spec,
                               "target": target_str_sequence, "situation": situation_spec,
                               "attention_weights_input": attention_weights_commands,
                               "attention_weights_situation": attention_weights_situations,
                               "accuracy": accuracy,
                               "exact_match": True if accuracy == 100 else False})
        json.dump(output, outfile, indent=4)
    return output_file_path


def predict(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int) -> torch.Tensor:
    """
    TODO
    :param data_iterator:
    :param model:
    :param max_decoding_steps:
    :param pad_idx:
    :param sos_idx:
    :param eos_idx:
    :return:
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths) in data_iterator:

        # Encode the input sequence.
        encoded_input = model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)

        # Iteratively decode the output.
        output_sequence = []
        hidden = model.attention_decoder.initialize_hidden(encoded_input["hidden_states"])
        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []
        while token != eos_idx and decoding_iteration <= max_decoding_steps:
            output, hidden, attention_weights_command, attention_weights_situation = model.decode_input(
                target_token=token, hidden=hidden, encoder_outputs=encoded_input["encoded_commands"]["encoder_outputs"],
                input_lengths=input_lengths, encoded_situations=encoded_input["encoded_situations"])
            output = F.log_softmax(output, dim=-1)
            token = output.max(dim=-1)[1]
            output_sequence.append(token.data[0].item())
            attention_weights_commands.append(attention_weights_command.tolist())
            attention_weights_situations.append(attention_weights_situation.tolist())
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            attention_weights_commands.pop()
            attention_weights_situations.pop()
        yield (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
               attention_weights_commands, attention_weights_situations)

    elapsed_time = time.time() - start_time
    logging.info("Done predicting in {} seconds.".format(elapsed_time))
