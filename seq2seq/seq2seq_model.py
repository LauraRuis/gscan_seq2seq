# Code adapted from:
#   https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import logging
from typing import List
from typing import Tuple

from seq2seq.helpers import sequence_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class EncoderRNN(nn.Module):
    """
    Embed a sequence of symbols using an LSTM.

    The RNN hidden vector (not cell vector) at each step is captured,
      for transfer to an attention-based decoder
    """
    def __init__(self, input_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_probability: float, bidirectional: bool, padding_idx: int):
        """
        :param input_size: number of input symbols
        :param embedding_dim: number of hidden units in RNN encoder, and size of all embeddings
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        :param bidirectional: use a bidirectional LSTM instead and sum of the resulting embeddings
        """
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.dropout_probability = dropout_probability
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout_probability, bidirectional=bidirectional)

    def forward(self, input_batch: torch.LongTensor, input_lengths: List[int]) -> Tuple[torch.Tensor, dict]:
        """
        :param input_batch: [batch_size, max_length]; batched padded input sequences
        :param input_lengths: length of each padded input sequence.
        :return: hidden states for last layer of last time step, the output of the last layer per time step and
        the sequence lengths per example in the batch.
        NB: The hidden states in the bidirectional case represent the final hidden state of each directional encoder,
        meaning the whole sequence in both directions, whereas the output per time step represents different parts of
        the sequences (0:t for the forward LSTM, t:T for the backward LSTM).
        """
        assert input_batch.size(0) == len(input_lengths), "Wrong amount of lengths passed to .forward()"
        input_embeddings = self.embedding(input_batch)  # [batch_size, max_length, embedding_dim]
        input_embeddings = self.dropout(input_embeddings)  # [batch_size, max_length, embedding_dim]

        # Sort the sequences by length in descending order.
        batch_size = len(input_lengths)
        max_length = max(input_lengths)
        input_lengths = torch.tensor(input_lengths, device=device, dtype=torch.long)
        input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        input_embeddings = input_embeddings[perm_idx]  # TODO: autograd safe?  torch.index_select

        # RNN embedding.
        packed_input = pack_padded_sequence(input_embeddings, input_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        # hidden, cell [num_layers * num_directions, batch_size, embedding_dim]
        # hidden and cell are unpacked, such that they store the last hidden state for each sequence in the batch.
        output_per_timestep, _ = pad_packed_sequence(
            packed_output)  # [max_length, batch_size, embedding_dim * num_directions]

        # If biLSTM, sum the outputs for each direction
        # TODO: ask question about this (why not half the hidden size and concat for example?)
        if self.bidirectional:
            output_per_timestep = output_per_timestep.view(max_length, batch_size, 2, self.embedding_dim)
            output_per_timestep = torch.sum(output_per_timestep, 2)  # [max_length, batch_size, embedding_dim]
            hidden = hidden.view(self.num_layers, 2, batch_size, self.embedding_dim)
            hidden = torch.sum(hidden, 1)  # [num_layers, batch_size, embedding_dim]
        hidden = hidden[-1, :, :]  # [batch_size, embedding_dim] (get the last layer)

        # Reverse the sorting.
        _, unperm_idx = perm_idx.sort(0)
        hidden = hidden[unperm_idx, :]
        output_per_timestep = output_per_timestep[:, unperm_idx, :]
        input_lengths = input_lengths[unperm_idx].tolist()

        return hidden, {"encoder_outputs": output_per_timestep, "sequence_lengths": input_lengths}

    def extra_repr(self) -> str:
        return "EncoderRNN\n bidirectional={} \n num_layers={}\n hidden_size={}\n dropout={}\n "\
               "n_input_symbols={}\n".format(self.bidirectional, self.num_lauers, self.hidden_size,
                                             self.dropout_probability, self.input_size)


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                                torch.Tensor]:
        """
        :param queries: [batch_size, num_queries, query_dim]
        :param keys: [batch_size, num_memory, query_dim]
        :param values: [batch_size, num_memory, value_dim]
        :return: soft-retrieval of values; [batch_size, num_queries, value_dim]
         attention_weights : [batch_size, num_queries, num_memory]
        """
        # [bsz, num_queries, query_dim] X [bsz, query_dim, num_memory] = [bsz, num_queries, num_memory]
        attention_weights = torch.bmm(queries, keys.transpose(1, 2))
        attention_weights = F.softmax(attention_weights, dim=2)  # [batch_size, num_queries, num_memory]

        # [bsz, num_queries, num_memory] X [bsz, num_memory, value_dim] = [bsz, num_queries, value_dim]
        soft_values_retrieval = torch.bmm(attention_weights, values)
        return soft_values_retrieval, attention_weights

    @staticmethod
    def forward_masked(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, memory_lengths: List[int]):
        """
        Key-value memory which takes queries and retrieves weighted combinations of values
          This version masks out certain memories, so that you can differing numbers of memories per batch.

        :param queries: [batch_size, num_queries, query_dim]
        :param keys: [batch_size, num_memory, query_dim]
        :param values: [batch_size, num_memory, value_dim]
        :param memory_lengths: [batch_size] actual number of keys in each batch
        :return:
            soft_values_retrieval : soft-retrieval of values; [batch_size, n_queries, value_dim]
            attention_weights : soft-retrieval of values; [batch_size, n_queries, n_memory]
        """
        query_dim = torch.tensor(queries.size(2), dtype=torch.float, device=device)
        batch_size = keys.size(0)
        assert len(memory_lengths) == batch_size
        memory_lengths = torch.tensor(memory_lengths, dtype=torch.long, device=device)

        # [bsz, num_queries, query_dim] X [bsz, query_dim, num_memory] = [bsz, num_queries, num_memory]
        attention_weights = torch.bmm(queries, keys.transpose(1, 2))
        attention_weights = torch.div(attention_weights, torch.sqrt(query_dim))

        # Mask out keys that are on a padding location.
        mask = sequence_mask(memory_lengths)  # [batch_size, num_memory]
        mask = mask.unsqueeze(1).expand_as(attention_weights)  # [batch_size, num_queries, num_memory]
        attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))  # fill with large negative numbers
        attention_weights = F.softmax(attention_weights, dim=2)  # [batch_size, num_queries, num_memory]

        # [bsz, num_queries, num_memory] X [bsz, num_memory, value_dim] = [bsz, num_queries, value_dim]
        soft_values_retrieval = torch.bmm(attention_weights, values)
        return soft_values_retrieval, attention_weights


class AttentionDecoderRNN(nn.Module):
    """One-step batch decoder with Luong et al. attention"""

    def __init__(self, hidden_size: int, output_size: int, num_layers: int, dropout_probability=0.1):
        """
        :param hidden_size: number of hidden units in RNN, and embedding size for output symbols
        :param output_size: number of output symbols
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        """
        super(AttentionDecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)  # TODO: change
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_probability)
        self.attention = Attention()
        self.hidden_context_to_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward_step(self, input_tokens: torch.LongTensor, last_hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoder_outputs: torch.Tensor, encoder_lengths: List[int]) -> Tuple[torch.Tensor,
                                                                                         Tuple[torch.Tensor,
                                                                                               torch.Tensor],
                                                                                         torch.Tensor]:
        """
        Run batch decoder forward for a single time step.
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param input_tokens: one time step inputs tokens of length batch_size
        :param last_hidden: previous decoder state, which is pair of tensors [num_layers, batch_size, hidden_size]
        (pair for hidden and cell)
        :param encoder_outputs: all encoder outputs, [max_input_length, batch_size, embedding_dim]
        :param encoder_lengths: length of each padded input sequence that were passed to the encoder.
        :return: output : un-normalized output probabilities, [batch_size, output_size]
          hidden : current decoder state, which is a pair of tensors [num_layers, batch_size, hidden_size]
           (pair for hidden and cell)
          attention_weights : attention weights, [batch_size, 1, max_input_length]
        """

        # Embed each input symbol
        embedded_input = self.embedding(input_tokens)  # [batch_size, hidden_size]
        embedded_input = self.dropout(embedded_input)
        embedded_input = embedded_input.unsqueeze(0)  # [1, batch_size, hidden_size]

        lstm_output, hidden = self.lstm(embedded_input, last_hidden)
        # lstm_output: [1, batch_size, hidden_size]
        # hidden: tuple of each [num_layers, batch_size, hidden_size] (pair for hidden and cell)
        # TODO: check attention over padding locations
        context, attention_weights = self.attention.forward_masked(lstm_output.transpose(0, 1),
                                                                   encoder_outputs.transpose(0, 1),
                                                                   encoder_outputs.transpose(0, 1),
                                                                   memory_lengths=encoder_lengths)
        # context : [batch_size, 1, hidden_size]
        # attention_weights : [batch_size, 1, max_input_length]

        # Concatenate the context vector and RNN hidden state, and map to an output
        lstm_output = lstm_output.squeeze(0)  # [batch_size, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]
        attention_weights = attention_weights.squeeze(1)  # [batch_size, max_input_length]
        concat_input = torch.cat([lstm_output, context], dim=1)  # [batch_size, hidden_size*2]
        concat_output = self.tanh(self.hidden_context_to_hidden(concat_input))  # [batch_size, hidden_size]
        output = self.hidden_to_output(concat_output)  # [batch_size, output_size]
        return output, hidden, attention_weights
        # output : [un-normalized probabilities] [batch_size, output_size]
        # hidden: tuple of size [num_layers, batch_size, hidden_size] (for hidden and cell)
        # attention_weights: [batch_size, max_input_length]

    def forward(self, input_tokens: torch.LongTensor, input_lengths: List[int],
                init_hidden: Tuple[torch.Tensor, torch.Tensor], encoder_outputs: torch.Tensor,
                encoder_lengths: List[int]):
        """
        Run batch attention decoder forward for a series of steps
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param input_tokens: [batch_size, max_length];  padded target sequences
        :param input_lengths: [batch_size] for sequence length of each padded target sequence
        :param init_hidden: tuple of tensors [num_layers, batch_size, hidden_size] (for hidden and cell)
        :param encoder_outputs: [max_input_length, batch_size, embedding_dim]
        :param encoder_lengths: [batch_size] sequence length of each encoder sequence (without padding)
        :return: output : unnormalized log-score, [max_length, batch_size, output_size]
          hidden : current decoder state, tuple with each [num_layers, batch_size, hidden_size] (for hidden and cell)
        """
        input_embedded = self.embedding(input_tokens)  # [batch_size, max_length, embedding_dim]
        input_embedded = self.dropout(input_embedded)  # [batch_size, max_length, embedding_dim]

        # Sort the sequences by length in descending order
        input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
        input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        input_embedded = input_embedded[perm_idx]
        initial_h, initial_c = init_hidden
        init_hidden = (initial_h[:, perm_idx, :], initial_c[:, perm_idx, :])

        # RNN decoder
        packed_input = pack_padded_sequence(input_embedded, input_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_input, init_hidden)
        # hidden is [num_layers, batch_size, hidden_size] (pair for hidden and cell)
        lstm_output, _ = pad_packed_sequence(packed_output)  # [max_length, batch_size, hidden_size]

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = lstm_output[:, unperm_idx, :]  # [max_length, batch_size, hidden_size]
        seq_len = input_lengths[unperm_idx].tolist()

        # Compute context vector using attention  TODO: ask why attention after lstm instead of before
        context, attention_weights = self.attention.forward_masked(lstm_output.transpose(0, 1),
                                                                   encoder_outputs.transpose(0, 1),
                                                                   encoder_outputs.transpose(0, 1),
                                                                   memory_lengths=encoder_lengths)
        # context: [batch_size, max_length, hidden_size]
        # attention_weights: [batch_size, max_length, max_input_length]

        # Concatenate the context vector and RNN hidden state, and map to an output
        concat_input = torch.cat([lstm_output, context.transpose(0, 1)], 2)  # [max_length, batch_size, hidden_size*2]
        concat_output = self.tanh(self.hidden_context_to_hidden(concat_input))  # [max_length, batch_size, hidden_size]
        output = self.hidden_to_output(concat_output)  # [max_length, batch_size, output_size]
        return output, seq_len
        # output : [unnormalized log-score] [max_length, batch_size, output_size]
        # seq_len : length of each output sequence

    def initialize_hidden(self, encoder_message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Populate the hidden variables with a message from the encoder.
        All layers, and both the hidden and cell vectors, are filled with the same message.
        :param encoder_message:  [batch_size, hidden_size] tensor
        :return: tuple of Tensors representing the hidden and cell state of shape: [num_layers, batch_size, hidden_dim]
        """
        encoder_message = encoder_message.unsqueeze(0)  # [1, batch_size, hidden_size]
        encoder_message = encoder_message.expand(self.num_layers, -1,
                                                 -1).contiguous()  # [num_layers, batch_size, hidden_size]
        return encoder_message.clone(), encoder_message.clone()

    def extra_repr(self) -> str:
        return "AttentionDecoderRNN\n num_layers={}\n hidden_size={}\n dropout={}\n num_output_symbols={}\n".format(
            self.num_layers, self.hidden_size, self.dropout_probability, self.output_size
        )


class DecoderRNN(nn.Module):
    """One-step simple batch RNN decoder"""

    def __init__(self, hidden_size: int, output_size: int, num_layers: int, dropout_probability=0.1):
        """
        :param hidden_size: number of hidden units in RNN, and embedding size for output symbols
        :param output_size: number of output symbols
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        """
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_probability)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, input_tokens: torch.LongTensor, last_hidden: torch.Tensor):
        """
        Run batch decoder forward for a single time step.

        :param input_tokens: [batch_size]
        :param last_hidden: previous decoder state, tuple of [num_layers, batch_size, hidden_size] (for hidden and cell)
        :return:
          output : un-normalized output probabilities, [batch_size, output_size]
          hidden : current decoder state, tuple of [num_layers, batch_size, hidden_size] (for hidden and cell)
        """

        # Embed each input symbol
        embedding = self.embedding(input_tokens)  # [batch_size, hidden_size]
        embedding = self.dropout(embedding)
        embedding = embedding.unsqueeze(0)  # [1, batch_size, hidden_size]
        lstm_output, hidden = self.lstm(embedding, last_hidden)
        # rnn_output is [1, batch_size, hidden_size]
        # hidden is [num_layers, batch_size, hidden_size] (pair for hidden and cell)
        lstm_output = lstm_output.squeeze(0)  # [batch_size, hidden_size]
        output = self.hidden_to_output(lstm_output)  # [batch_size, output_size]
        return output, hidden
        # output : un-normalized probabilities [batch_size, output_size]
        # hidden: pair of size [num_layers, batch_size, hidden_size] (for hidden and cell)

    def init_hidden(self, encoder_message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Populate the hidden variables with a message from the decoder.
        All layers, and both the hidden and cell vectors, are filled with the same message.
        :param encoder_message: [batch_size, hidden_size]
        :return:
        """
        encoder_message = encoder_message.unsqueeze(0)  # 1, batch_size, hidden_size
        encoder_message = encoder_message.expand(self.num_layers, -1,
                                                 -1).contiguous()  # nlayers, batch_size, hidden_size tensor
        return encoder_message.clone(), encoder_message.clone()

    def extra_repr(self) -> str:
        return "DecoderRNN\n num_layers={}\n hidden_size={}\n dropout={}\n num_output_symbols={}\n".format(
            self.num_layers, self.hidden_size, self.dropout_probability, self.output_size
        )
