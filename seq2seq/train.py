# TODO: implement training loop
import logging
import torch

from seq2seq.seq2seq_model import EncoderRNN
from seq2seq.gSCAN_dataset import GroundedScanDataset

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


def train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, embedding_dim: int, num_encoder_layers: int, encoder_dropout_p: float,
          encoder_bidirectional: bool, training_batch_size: int,
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

    data_batch, batch_lengths = training_set.get_data_batch(batch_size=training_batch_size)

    hidden, _ = encoder(data_batch, batch_lengths)
    print()
