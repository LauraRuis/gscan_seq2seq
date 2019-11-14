# TODO: implement training loop
import logging
import torch

from seq2seq.seq2seq_model import EncoderRNN
from seq2seq.seq2seq_model import DecoderRNN
from seq2seq.seq2seq_model import AttentionDecoderRNN
from seq2seq.gSCAN_dataset import GroundedScanDataset

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


def train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, embedding_dim: int, num_encoder_layers: int, encoder_dropout_p: float,
          encoder_bidirectional: bool, training_batch_size: int, num_decoder_layers: int, decoder_dropout_p: float,
          seed=42, **kwargs):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
    cfg = locals().copy()

    torch.manual_seed(seed)

    logger.info("Loading Training set...")
    training_set = GroundedScanDataset(data_path, data_directory, split="train",
                                       input_vocabulary_file=input_vocab_path,
                                       target_vocabulary_file=target_vocab_path,
                                       generate_vocabulary=generate_vocabularies)
    logger.info("Done Loading Training set.")
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))

    test_sentence = ["This", "is", "a", "circle"]
    array = training_set.sentence_to_array(test_sentence, vocabulary="input")
    test_sentence_decoded = training_set.array_to_sentence(array, vocabulary="input")
    if generate_vocabularies:
        training_set.save_vocabularies(input_vocab_path, target_vocab_path)

    # TESTING
    encoder = EncoderRNN(input_size=training_set.input_vocabulary_size, embedding_dim=embedding_dim,
                         num_layers=num_encoder_layers, dropout_probability=encoder_dropout_p,
                         bidirectional=encoder_bidirectional)
    attention_decoder = AttentionDecoderRNN(hidden_size=embedding_dim, output_size=training_set.target_vocabulary_size,
                                            num_layers=num_decoder_layers, dropout_probability=decoder_dropout_p)

    input_batch, input_lengths, target_batch, target_lengths = training_set.get_data_batch(
        batch_size=training_batch_size)

    hidden, encoder_outputs = encoder(input_batch, input_lengths)
    max_time = max(target_lengths)
    initial_hidden = attention_decoder.initialize_hidden(hidden)
    # TODO: check that hidden and cell aren't the same now (because of mutable type)
    decoder_outputs = []
    hidden = initial_hidden[0].clone(), initial_hidden[1].clone()
    for t in range(max_time):
        input_tokens = target_batch[:, t]
        output, hidden, attention_weights = attention_decoder.forward_step(input_tokens, hidden,
                                                                           encoder_outputs["encoder_outputs"].clone(),
                                                                           encoder_outputs["sequence_lengths"])
        decoder_outputs.append(output.unsqueeze(0))
    decoder_output = torch.cat(decoder_outputs, dim=0)  # [max_target_length, batch_size, output_vocabulary_size]
    # initial_hidden = initial_hidden[0].clone(), initial_hidden[1].clone()
    # decoder_output_batched = attention_decoder(target_batch, target_lengths, initial_hidden,
    #                                            encoder_outputs["encoder_outputs"].clone(),
    #                                            encoder_outputs["sequence_lengths"])
    print()
