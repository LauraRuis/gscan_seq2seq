# TODO: wrapper model class for input, situation -> output
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List
from typing import Dict
from typing import Tuple
import os
import shutil

from seq2seq.cnn_model import ConvolutionalNet
from seq2seq.seq2seq_model import EncoderRNN
from seq2seq.seq2seq_model import AttentionDecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


class Model(nn.Module):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, target_vocabulary_size: int, encoder_dropout_p: float,
                 encoder_bidirectional: bool, num_decoder_layers: int, decoder_dropout_p: float, image_dimensions: int,
                 num_cnn_channels: int, cnn_kernel_size: int, cnn_dropout_p: float, cnn_hidden_num_channels: int,
                 cnn_hidden_size: int, input_padding_idx: int, max_pool_kernel_size: int, max_pool_stride: int,
                 target_pad_idx: int, target_eos_idx: int, output_directory: str,
                 **kwargs):
        super(Model, self).__init__()

        # TODO: check that all these also get dropout_p = 0 if model.test()
        self.encoder = EncoderRNN(input_size=input_vocabulary_size, embedding_dim=embedding_dimension,
                                  hidden_size=encoder_hidden_size, num_layers=num_encoder_layers,
                                  dropout_probability=encoder_dropout_p, bidirectional=encoder_bidirectional,
                                  padding_idx=input_padding_idx)

        self.attention_decoder = AttentionDecoderRNN(hidden_size=embedding_dimension,
                                                     output_size=target_vocabulary_size, num_layers=num_decoder_layers,
                                                     dropout_probability=decoder_dropout_p)

        self.situation_encoder = ConvolutionalNet(image_width=image_dimensions, num_channels=num_cnn_channels,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  kernel_size=cnn_kernel_size, dropout_probability=cnn_dropout_p,
                                                  max_pool_kernel_size=max_pool_kernel_size,
                                                  max_pool_stride=max_pool_stride,
                                                  intermediate_hidden_size=cnn_hidden_size,
                                                  output_dimension=embedding_dimension)

        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.loss_criterion = nn.NLLLoss(ignore_index=target_pad_idx)

        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0

    @staticmethod
    def remove_start_of_sequence(input_tensor: torch.Tensor) -> torch.Tensor:
        # Get rid of start-of-sequence-tokens in targets batch and append a padding token to each example in the batch.
        batch_size, max_time = input_tensor.size()
        input_tensor = input_tensor[:, 1:]
        input_tensor = torch.cat([input_tensor, torch.zeros(batch_size, dtype=torch.long).unsqueeze(dim=1)], dim=1)
        return input_tensor

    def get_accuracy(self, target_scores: torch.Tensor, targets: torch.Tensor) -> float:
        """
        TODO: write test for accuracy
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            targets = self.remove_start_of_sequence(targets)
            mask = (targets != self.target_pad_idx).long()
            total = mask.sum().data.item()
            predicted_targets = target_scores.max(dim=2)[1]
            equal_targets = torch.eq(targets.data, predicted_targets.data).long()
            match_targets = (equal_targets * mask).sum().data.item()
            accuracy = 100. * match_targets / total
        return accuracy

    def get_loss(self, target_scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        TODO: write test for loss
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets: ground-truth targets of size [batch_size, max_target_length]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        targets = self.remove_start_of_sequence(targets)

        # Calculate the loss.
        _, _, vocabulary_size = target_scores.size()
        target_scores_2d = target_scores.reshape(-1, vocabulary_size)
        loss = self.loss_criterion(target_scores_2d, targets.view(-1))
        return loss

    def encode_input(self, commands_input: torch.LongTensor, commands_lengths: List[int],
                     situations_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded_image = self.situation_encoder(situations_input)
        hidden, encoder_outputs = self.encoder(commands_input, commands_lengths)
        return {"encoded_situations": encoded_image, "encoded_commands": encoder_outputs, "hidden_states": hidden}

    def predict_sequence(self):
        raise NotImplementedError()

    def decode_input(self, target_token: torch.LongTensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                     input_lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, hidden, attention_weights = self.attention_decoder.forward_step(target_token, hidden,
                                                                                encoder_outputs, input_lengths)
        return output, hidden, attention_weights

    def decode_input_batched(self, target_batch: torch.LongTensor, target_lengths: List[int],
                             initial_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                             input_lengths: List[int]) -> torch.Tensor:
        initial_hidden = self.attention_decoder.initialize_hidden(initial_hidden)
        decoder_output_batched, _ = self.attention_decoder(target_batch, target_lengths, initial_hidden,
                                                           encoder_outputs, input_lengths)
        decoder_output_batched = F.log_softmax(decoder_output_batched, dim=-1)
        return decoder_output_batched

    def forward(self, commands_input: torch.LongTensor, commands_lengths: List[int], situations_input: torch.Tensor,
                target_batch: torch.LongTensor, target_lengths: List[int]):

        encoder_output = self.encode_input(commands_input=commands_input, commands_lengths=commands_lengths,
                                           situations_input=situations_input)
        decoder_output = self.decode_input_batched(
            target_batch=target_batch, target_lengths=target_lengths, initial_hidden=encoder_output["hidden_states"],
            encoder_outputs=encoder_output["encoded_commands"]["encoder_outputs"], input_lengths=commands_lengths)
        return decoder_output.transpose(0, 1)  # [batch_size, max_target_seq_length, target_vocabulary_size]

    def update_state(self, is_best: bool) -> {}:
        # TODO: also save best loss and load
        self.trained_iterations += 1
        if is_best:
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        # TODO: also save best loss and load
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        # TODO: also save best loss and load
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path
