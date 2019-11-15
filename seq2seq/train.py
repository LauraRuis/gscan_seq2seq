# TODO: implement training loop
import logging
import torch

from seq2seq.model import Model
from seq2seq.gSCAN_dataset import GroundedScanDataset

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


def train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, embedding_dimension: int, num_encoder_layers: int, encoder_dropout_p: float,
          encoder_bidirectional: bool, training_batch_size: int, num_decoder_layers: int, decoder_dropout_p: float,
          cnn_kernel_size: int, cnn_dropout_p: float, cnn_hidden_num_channels: int, max_pool_kernel_size: int,
          encoder_hidden_size: int, max_pool_stride: int, cnn_hidden_size: int, seed=42, **kwargs):
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

    input_batch, input_lengths, target_batch, target_lengths, situation_batch = training_set.get_data_batch(
        batch_size=training_batch_size)
    image_dimensions, _, num_channels = situation_batch[0].shape

    model = Model(image_dimensions=image_dimensions, input_vocabulary_size=training_set.input_vocabulary_size,
                  target_vocabulary_size=training_set.target_vocabulary_size, num_cnn_channels=num_channels,
                  input_padding_idx=training_set.input_vocabulary.pad_idx, **cfg)

    test = model(input_batch, input_lengths, situation_batch, target_batch, target_lengths)

    print()
