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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def describe_model(net):
    if type(net) is AttnDecoderRNN:
        logger.info('AttnDecoderRNN specs:')
        logger.info(' nlayers=' + str(net.num_layers))
        logger.info(' hidden_size=' + str(net.hidden_size))
        logger.info(' dropout=' + str(net.dropout_p))
        logger.info(' n_output_symbols=' + str(net.output_size))
        logger.info('')
    elif type(net) is EncoderRNN:
        logger.info('EncoderRNN specs:')
        logger.info(' bidirectional=' + str(net.bi))
        logger.info(' nlayers=' + str(net.num_layers))
        logger.info(' hidden_size=' + str(net.embedding_dim))
        logger.info(' dropout=' + str(net.dropout_p))
        logger.info(' n_input_symbols=' + str(net.input_size))
        logger.info('')
    elif type(net) is DecoderRNN:
        logger.info('DecoderRNN specs:')
        logger.info(' nlayers=' + str(net.num_layers))
        logger.info(' hidden_size=' + str(net.hidden_size))
        logger.info(' dropout=' + str(net.dropout_p))
        logger.info(' n_output_symbols=' + str(net.output_size))
        logger.info("")
    else:
        logger.info('Network type not found...')


class EncoderRNN(nn.Module):
    """
    Embed a sequence of symbols using an LSTM.

    The RNN hidden vector (not cell vector) at each step is captured,
      for transfer to an attention-based decoder
    # TODO: hidden size = embedding dim in nn.LSTM?
    """
    def __init__(self, input_size: int, embedding_dim: int, num_layers: int, dropout_probability: float,
                 bidirectional: bool):
        """
        :param input_size: number of input symbols
        :param embedding_dim: number of hidden units in RNN encoder, and size of all embeddings
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        :param bidirectional: use a bidirectional LSTM instead and sum of the resulting embeddings
        """
        super(EncoderRNN, self).__init__()
        # TODO: separate hidden dim instead of embedding dim
        self.num_layers = num_layers
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.dropout_probability = dropout_probability
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=num_layers,
                            dropout=dropout_probability, bidirectional=bidirectional)

    def forward(self, input_batch: torch.LongTensor, input_lengths: List[int]) -> Tuple[torch.Tensor, dict]:
        """
        :param input_batch: (batch_size x max_length); batched padded input sequences
        :param input_lengths: length of each padded input sequence.
        :return: ... TODO
        """
        assert input_batch.size(0) == len(input_lengths), "Wrong amount of lengths passed to .forward()"
        input_embeddings = self.embedding(input_batch)  # [batch_size, max_length, embedding_size]
        input_embeddings = self.dropout(input_embeddings)  # [batch_size, max_length, embedding_size]

        # Sort the sequences by length in descending order.
        batch_size = len(input_lengths)
        max_length = max(input_lengths)
        input_lengths = torch.tensor(input_lengths, device=device, dtype=torch.long)
        input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        input_embeddings = input_embeddings[perm_idx]  # TODO: autograd safe?  torch.index_select

        # RNN embedding
        packed_input = pack_padded_sequence(input_embeddings, input_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        # hidden, cell [num_layers * num_directions, batch_size, embedding_size]
        # hidden and cell are unpacked, such that they store the last hidden state for each unpadded sequence
        output_per_timestep, _ = pad_packed_sequence(
            packed_output)  # [max_length, batch_size, embedding_size * num_directions]

        # If biLSTM, sum the outputs for each direction
        # TODO: ask question about this (why not half the hidden size and concat for example?)
        if self.bidirectional:  # TODO: why isn't hidden per timestep of last timestep the same as hidden of last layer
            output_per_timestep = output_per_timestep.view(max_length, batch_size, 2, self.embedding_dim)
            output_per_timestep = torch.sum(output_per_timestep, 2)  # [max_length, batch_size, embedding_size]
            hidden = hidden.view(self.num_layers, 2, batch_size, self.embedding_dim)
            hidden = torch.sum(hidden, 1)  # [num_layers, batch_size, embedding_size]
        hidden = hidden[-1, :, :]  # n, embedding_size (grab the last layer)

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        hidden = hidden[unperm_idx, :]  # n, embedding_size
        output_per_timestep = output_per_timestep[:, unperm_idx, :]  # max_length, batch_size, embedding_size
        seq_len = input_lengths[unperm_idx].tolist()

        return hidden, {"embed_by_step": output_per_timestep, "seq_len": seq_len}
        # hidden is (n, embedding_size);
        # embed_by_step is (max_length, n, embedding_size)
        # seq_len is tensor of length n


class Attn(nn.Module):

    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, Q, K, V):
        #
        # Input
        #  Q : Matrix of queries; batch_size x n_queries x query_dim
        #  K : Matrix of keys; batch_size x n_memory x query_dim
        #  V : Matrix of values; batch_size x n_memory x value_dim
        #
        # Output
        #  R : soft-retrieval of values; batch_size x n_queries x value_dim
        #  attn_weights : soft-retrieval of values; batch_size x n_queries x n_memory
        query_dim = torch.tensor(float(Q.size(2)))
        if Q.is_cuda: query_dim = query_dim.cuda()
        attn_weights = torch.bmm(Q, K.transpose(1, 2))  # batch_size x n_queries x n_memory
        # attn_weights = torch.div(attn_weights, torch.sqrt(query_dim))
        attn_weights = F.softmax(attn_weights, dim=2)  # batch_size x n_queries x n_memory
        R = torch.bmm(attn_weights, V)  # batch_size x n_queries x value_dim
        return R, attn_weights


class AttnDecoderRNN(nn.Module):

    # One-step batch decoder with Luong et al. attention
    def __init__(self, hidden_size, output_size, nlayers, dropout_p=0.1):
        # Input
        #  hidden_size : number of hidden units in RNN, and embedding size for output symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers
        #  dropout_p : dropout applied to symbol embeddings and RNNs
        super(AttnDecoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=nlayers, dropout=dropout_p)
        self.attn = Attn()
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Run batch decoder forward for a single time step.
        #  Each decoder step considers all of the encoder_outputs through attention.
        #  Attention retrieval is based on decoder hidden state (not cell state)
        #
        # Input
        #  input: LongTensor of length batch_size
        #  last_hidden: previous decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #  encoder_outputs: all encoder outputs, max_input_length x batch_size x embedding_size
        #
        # Output
        #   output : unnormalized output probabilities, batch_size x output_size
        #   hidden : current decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #   attn_weights : attention weights, batch_size x 1 x max_input_length

        # Embed each input symbol
        batch_size = input.numel()
        embedding = self.embedding(input)  # batch_size x hidden_size
        embedding = self.dropout(embedding)
        embedding = embedding.unsqueeze(0)  # S=1 x batch_size x hidden_size

        rnn_output, hidden = self.rnn(embedding, last_hidden)
        # rnn_output is S=1 x batch_size x hidden_size
        # hidden is nlayer x batch_size x hidden_size (pair for hidden and cell)

        context, attn_weights = self.attn(rnn_output.transpose(0, 1), encoder_outputs.transpose(0, 1),
                                          encoder_outputs.transpose(0, 1))
        # context : batch_size x 1 x hidden_size
        # attn_weights : batch_size x 1 x max_input_length

        # Concatenate the context vector and RNN hidden state, and map to an output
        rnn_output = rnn_output.squeeze(0)  # batch_size x hidden_size
        context = context.squeeze(1)  # batch_size x hidden_size
        attn_weights = attn_weights.squeeze(1)  # batch_size x max_input_length
        concat_input = torch.cat((rnn_output, context), 1)  # batch_size x hidden_size*2
        concat_output = self.tanh(self.concat(concat_input))  # batch_size x hidden_size
        output = self.out(concat_output)  # batch_size x output_size
        return output, hidden, attn_weights
        # output : [unnormalized probabilities] batch_size x output_size
        # hidden: pair of size [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        # attn_weights: batch_size x max_input_length

    def initHidden(self, encoder_message):
        # Populate the hidden variables with a message from the decoder.
        # All layers, and both the hidden and cell vectors, are filled with the same message.
        #   message : batch_size x hidden_size tensor
        encoder_message = encoder_message.unsqueeze(0)  # 1 x batch_size x hidden_size
        encoder_message = encoder_message.expand(self.nlayers, -1,
                                                 -1).contiguous()  # nlayers x batch_size x hidden_size tensor
        return (encoder_message, encoder_message)


class DecoderRNN(nn.Module):

    # One-step simple batch RNN decoder
    def __init__(self, hidden_size, output_size, nlayers, dropout_p=0.1):
        # Input
        #  hidden_size : number of hidden units in RNN, and embedding size for output symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers
        #  dropout_p : dropout applied to symbol embeddings and RNNs
        super(DecoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=nlayers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, last_hidden):
        # Run batch decoder forward for a single time step.
        #
        # Input
        #  input: LongTensor of length batch_size
        #  last_hidden: previous decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #
        # Output
        #   output : unnormalized output probabilities, batch_size x output_size
        #   hidden : current decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)

        # Embed each input symbol
        batch_size = input.numel()
        embedding = self.embedding(input)  # batch_size x hidden_size
        embedding = self.dropout(embedding)
        embedding = embedding.unsqueeze(0)  # S=1 x batch_size x hidden_size
        rnn_output, hidden = self.rnn(embedding, last_hidden)
        # rnn_output is S=1 x batch_size x hidden_size
        # hidden is nlayer x batch_size x hidden_size (pair for hidden and cell)
        rnn_output = rnn_output.squeeze(0)  # batch_size x hidden_size
        output = self.out(rnn_output)  # batch_size x output_size
        return output, hidden
        # output : [unnormalized probabilities] batch_size x output_size
        # hidden: pair of size [nlayer x batch_size x hidden_size] (pair for hidden and cell)

    def initHidden(self, encoder_message):
        # Populate the hidden variables with a message from the decoder.
        # All layers, and both the hidden and cell vectors, are filled with the same message.
        #   message : batch_size x hidden_size tensor
        encoder_message = encoder_message.unsqueeze(0)  # 1 x batch_size x hidden_size
        encoder_message = encoder_message.expand(self.nlayers, -1,
                                                 -1).contiguous()  # nlayers x batch_size x hidden_size tensor
        return (encoder_message, encoder_message)