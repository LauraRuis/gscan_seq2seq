# TODO: implement predict function
# TODO: move attention to before LSTM in seq2seq
# TODO: visualize training attention weights
# TODO: visualize training loss

import logging
import torch
import os

from seq2seq.model import Model
from seq2seq.gSCAN_dataset import GroundedScanDataset
from seq2seq.helpers import print_parameters
from seq2seq.evaluate import evaluate

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


def train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, embedding_dimension: int, num_encoder_layers: int, encoder_dropout_p: float,
          encoder_bidirectional: bool, training_batch_size: int, test_batch_size: int, max_decoding_steps: int,
          num_decoder_layers: int, decoder_dropout_p: float, situation_embedding_size: int,
          cnn_kernel_size: int, cnn_dropout_p: float, cnn_hidden_num_channels: int, max_pool_kernel_size: int,
          encoder_hidden_size: int, max_pool_stride: int, cnn_hidden_size: int, learning_rate: float,
          adam_beta_1: float, adam_beta_2: float, resume_from_file: str, max_training_iterations: int,
          output_directory: str, print_every: int, evaluate_every: int, max_training_examples=None,
          seed=42, **kwargs):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
    cfg = locals().copy()

    torch.manual_seed(seed)

    logger.info("Loading Training set...")
    training_set = GroundedScanDataset(data_path, data_directory, split="train",
                                       input_vocabulary_file=input_vocab_path,
                                       target_vocabulary_file=target_vocab_path,
                                       generate_vocabulary=generate_vocabularies)
    training_set.read_dataset(max_examples=max_training_examples)
    logger.info("Done Loading Training set.")
    logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))

    if generate_vocabularies:
        training_set.save_vocabularies(input_vocab_path, target_vocab_path)
        logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    logger.info("Loading Test set...")
    test_set = GroundedScanDataset(data_path, data_directory, split="test",  # TODO: also dev set
                                   input_vocabulary_file=input_vocab_path,
                                   target_vocabulary_file=target_vocab_path, generate_vocabulary=False)
    test_set.read_dataset(max_examples=kwargs["max_testing_examples"])
    logger.info("Done Loading Test set.")

    model = Model(image_dimensions=training_set.image_dimensions,
                  input_vocabulary_size=training_set.input_vocabulary_size,
                  target_vocabulary_size=training_set.target_vocabulary_size, num_cnn_channels=3,
                  input_padding_idx=training_set.input_vocabulary.pad_idx,
                  target_pad_idx=training_set.target_vocabulary.pad_idx,
                  target_eos_idx=training_set.target_vocabulary.eos_idx,
                  **cfg)
    model = model.cuda() if use_cuda else model
    print_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))

    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_iteration = 1
    best_accuracy = 0
    best_loss = float('inf')
    if resume_from_file:
        assert os.path.isfile(resume_from_file), "No checkpoint found at {}".format(resume_from_file)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file))
        optimizer_state_dict = model.load_model(resume_from_file)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(resume_from_file, start_iteration))

    # TODO: Make sure EOS doesn't get fed to the model, just needed in targets
    # TODO: Make sure SOS doesn't get taken into account for loss
    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < max_training_iterations:

        # Shuffle the dataset and loop over it.
        training_set.shuffle_data()
        for (input_batch, input_lengths, situation_batch, _, target_batch,
             target_lengths) in training_set.get_data_iterator(
                batch_size=training_batch_size):
            is_best = False
            model.train()
            target_scores = model(input_batch, input_lengths, situation_batch, target_batch, target_lengths)
            loss = model.get_loss(target_scores, target_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if training_iteration % print_every == 0:
                accuracy = model.get_accuracy(target_scores, target_batch)
                logger.info("Iteration %08d, loss %8.4f, accuracy %5.2f,learning_rate %.5f" % (training_iteration, loss,
                                                                                               accuracy, learning_rate))

            if training_iteration % evaluate_every == 0:
                model.eval()
                logger.info("Evaluating..")
                accuracy = evaluate(test_set.get_data_iterator(batch_size=1), model=model,
                                    max_decoding_steps=max_decoding_steps, sos_idx=test_set.target_vocabulary.sos_idx,
                                    eos_idx=test_set.target_vocabulary.eos_idx)
                logger.info("  Evaluation Accuracy: %5.2f" % accuracy)
                if accuracy > best_accuracy:
                    is_best = True
                    best_accuracy = accuracy
                    model.update_state(is_best=is_best)
                file_name = "checkpoint_it_{}.pth.tar".format(str(training_iteration))
                model.save_checkpoint(file_name=file_name, is_best=is_best, optimizer_state_dict=optimizer.state_dict())

            training_iteration += 1
            if training_iteration > max_training_iterations:
                break

    logger.info("Finished training.")

