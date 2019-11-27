# TODO: already make sequence masks in gSCAN_dataset.py
# TODO: visualize attention weights input command
# TODO: conditional attention
# TODO: check whether conv with filter size equal to grid size is necessary for performance
# TODO: learning rate schedule and stuff

import argparse
import logging
import os
import torch

from seq2seq.gSCAN_dataset import GroundedScanDataset
from seq2seq.model import Model
from seq2seq.train import train
from seq2seq.predict import predict_and_save

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt="%Y-%m-%d %H:%M")
logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


def main():

    parser = argparse.ArgumentParser(description="Sequence to sequence models for Grounded SCAN")

    # General arguments
    parser.add_argument("--mode", type=str, default="run_tests", help="train, test or predict", required=True)
    parser.add_argument("--output_directory", type=str, default="output")
    parser.add_argument("--resume_from_file", type=str, default="")

    # Data arguments
    parser.add_argument("--split", type=str, default="test", help="Which split to get from Grounded Scan.")
    parser.add_argument("--data_directory", type=str, default="data/uniform", help="Path to folder with data.")
    parser.add_argument("--input_vocab_path", type=str, default="training_input_vocab.txt",
                        help="Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
    parser.add_argument("--target_vocab_path", type=str, default="training_target_vocab.txt",
                        help="Path to file with target vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
    parser.add_argument("--generate_vocabularies", dest="generate_vocabularies", default=False, action="store_true")
    parser.add_argument("--load_vocabularies", dest="generate_vocabularies", default=True, action="store_false")

    # Training and learning arguments
    parser.add_argument("--training_batch_size", type=int, default=100)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--max_training_examples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_decay_steps', type=float, default=20000)
    parser.add_argument("--adam_beta_1", type=float, default=0.9)
    parser.add_argument("--adam_beta_2", type=float, default=0.999)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--evaluate_every", type=int, default=1000)
    parser.add_argument("--max_training_iterations", type=int, default=100000)

    # Testing and predicting arguments
    parser.add_argument("--max_testing_examples", type=int, default=None)
    parser.add_argument("--max_decoding_steps", type=int, default=25)
    parser.add_argument("--output_file_name", type=str, default="predict.json")

    # Situation Encoder arguments
    parser.add_argument("--simple_situation_representation", dest="simple_situation_representation", default=True,
                        action="store_true")
    parser.add_argument("--image_situation_representation", dest="simple_situation_representation", default=False,
                        action="store_false")
    parser.add_argument("--cnn_hidden_num_channels", type=int, default=100)
    parser.add_argument("--cnn_kernel_size", type=int, default=1)
    parser.add_argument("--cnn_dropout_p", type=float, default=0.3)

    # Command Encoder arguments
    parser.add_argument("--embedding_dimension", type=int, default=50)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--encoder_hidden_size", type=int, default=200)
    parser.add_argument("--encoder_dropout_p", type=float, default=0.3)
    parser.add_argument("--encoder_bidirectional", dest="encoder_bidirectional", default=True, action="store_true")
    parser.add_argument("--encoder_unidirectional", dest="encoder_bidirectional", default=False, action="store_false")

    # Decoder arguments
    parser.add_argument("--num_decoder_layers", type=int, default=1)
    parser.add_argument("--decoder_dropout_p", type=float, default=0.3)

    # Other arguments
    parser.add_argument("--seed", type=int, default=42)

    flags = vars(parser.parse_args())

    if not os.path.exists(flags["output_directory"]):
        os.mkdir(os.path.join(os.getcwd(), flags["output_directory"]))

    # Some checks on the flags
    if flags["generate_vocabularies"]:
        assert flags["input_vocab_path"] and flags["target_vocab_path"], "Please specify paths to vocabularies to save."

    if flags["test_batch_size"] > 1:
        raise NotImplementedError("Test batch size larger than 1 not implemented.")

    data_path = os.path.join(flags["data_directory"], "dataset.txt")
    if flags["mode"] == "train":
        train(data_path=data_path, **flags)
    elif flags["mode"] == "test":
        assert os.path.exists(os.path.join(flags["data_directory"], flags["input_vocab_path"])) and os.path.exists(
            os.path.join(flags["data_directory"], flags["target_vocab_path"])), \
            "No vocabs found at {} and {}".format(flags["input_vocab_path"], flags["target_vocab_path"])
        logger.info("Loading {} dataset split...".format(flags["split"]))
        test_set = GroundedScanDataset(data_path, flags["data_directory"], split=flags["split"],
                                       input_vocabulary_file=flags["input_vocab_path"],
                                       target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False)
        test_set.read_dataset(max_examples=flags["max_testing_examples"],
                              simple_situation_representation=flags["simple_situation_representation"])
        logger.info("Done Loading {} dataset split.".format(flags["split"]))
        logger.info("  Loaded {} examples.".format(test_set.num_examples))
        logger.info("  Input vocabulary size: {}".format(test_set.input_vocabulary_size))
        logger.info("  Most common input words: {}".format(test_set.input_vocabulary.most_common(5)))
        logger.info("  Output vocabulary size: {}".format(test_set.target_vocabulary_size))
        logger.info("  Most common target words: {}".format(test_set.target_vocabulary.most_common(5)))

        model = Model(image_dimensions=test_set.image_dimensions,
                      input_vocabulary_size=test_set.input_vocabulary_size,
                      target_vocabulary_size=test_set.target_vocabulary_size,
                      num_cnn_channels=test_set.image_channels,
                      input_padding_idx=test_set.input_vocabulary.pad_idx,
                      target_pad_idx=test_set.target_vocabulary.pad_idx,
                      target_eos_idx=test_set.target_vocabulary.eos_idx,
                      **flags)
        model = model.cuda() if use_cuda else model

        # Load model and vocabularies if resuming.
        assert os.path.isfile(flags["resume_from_file"]), "No checkpoint found at {}".format(flags["resume_from_file"])
        logger.info("Loading checkpoint from file at '{}'".format(flags["resume_from_file"]))
        model.load_model(flags["resume_from_file"])
        start_iteration = model.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(flags["resume_from_file"], start_iteration))
        output_file_path = os.path.join(flags["output_directory"], flags["output_file_name"])
        output_file = predict_and_save(dataset=test_set, model=model, output_file_path=output_file_path, **flags)
        logger.info("Saved predictions to {}".format(output_file))
    elif flags["mode"] == "predict":
        raise NotImplementedError()
    else:
        raise ValueError("Wrong value for parameters --mode ({}).".format(flags["mode"]))


if __name__ == "__main__":
    main()
