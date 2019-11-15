# TODO: wrapper model class for input, situation -> output
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List
from typing import Dict

from seq2seq.cnn_model import ConvolutionalNet
from seq2seq.seq2seq_model import EncoderRNN
from seq2seq.seq2seq_model import AttentionDecoderRNN

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


class Model(nn.Module):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, target_vocabulary_size: int, encoder_dropout_p: float,
                 encoder_bidirectional: bool, num_decoder_layers: int, decoder_dropout_p: float, image_dimensions: int,
                 num_cnn_channels: int, cnn_kernel_size: int, cnn_dropout_p: float, cnn_hidden_num_channels: int,
                 cnn_hidden_size: int, input_padding_idx: int, max_pool_kernel_size: int, max_pool_stride: int, **kwargs):
        super(Model, self).__init__()

        # TODO: check that all these also get dropout_p = 0 is model.test()
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

    def get_accuracy(self):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()

    def encode_input(self, commands_input: torch.LongTensor, commands_lengths: List[int],
                     situations_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded_image = self.situation_encoder(situations_input)
        hidden, encoder_outputs = self.encoder(commands_input, commands_lengths)
        return {"encoded_situations": encoded_image, "encoded_commands": encoder_outputs, "hidden_states": hidden}

    def predict_sequence(self):
        raise NotImplementedError()

    def decode_input(self, target_batch: torch.LongTensor, target_lengths: List[int], initial_hidden: torch.Tensor,
                     encoder_outputs: torch.Tensor, input_lengths: List[int]) -> torch.Tensor:
        initial_hidden = self.attention_decoder.initialize_hidden(initial_hidden)
        decoder_outputs = []
        max_time = max(target_lengths)
        hidden = initial_hidden
        for t in range(max_time):
            input_tokens = target_batch[:, t]
            output, hidden, attention_weights = self.attention_decoder.forward_step(input_tokens, hidden,
                                                                                    encoder_outputs, input_lengths)
            decoder_outputs.append(output.unsqueeze(0))
        decoder_output = torch.cat(decoder_outputs, dim=0)  # [max_target_length, batch_size, output_vocabulary_size]
        decoder_output = F.log_softmax(decoder_output, dim=-1)
        return decoder_output

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
