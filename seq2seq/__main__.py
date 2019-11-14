from seq2seq.gSCAN_dataset import GroundedScanDataset

import argparse
import logging

from seq2seq.train import train

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt="%Y-%m-%d %H:%M")
logger = logging.getLogger(__name__)


def main():

    parser = argparse.ArgumentParser(description="Sequence to sequence models for Grounded SCAN")

    # Data arguments
    parser.add_argument("--mode", type=str, default="train", help="train, test or predict")
    parser.add_argument("--data_directory", type=str, default="data", help="Path to folder with data.")
    parser.add_argument("--data_path", type=str, default="data/dataset.txt", help="Path to file with data.")
    parser.add_argument("--input_vocab_path", type=str, default="data/training_input_vocab.txt",
                        help="Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
    parser.add_argument("--target_vocab_path", type=str, default="data/training_target_vocab.txt",
                        help="Path to file with target vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
    parser.add_argument("--generate_vocabularies", dest="generate_vocabularies", default=False, action="store_true")
    parser.add_argument("--load_vocabularies", dest="generate_vocabularies", default=True, action="store_false")

    # Training arguments
    parser.add_argument('--training_batch_size', type=int, default=10)

    # Encoder arguments
    parser.add_argument('--embedding_dim', type=int, default=50)
    parser.add_argument('--num_encoder_layers', type=int, default=1)
    parser.add_argument('--encoder_dropout_p', type=float, default=0.1)
    parser.add_argument("--encoder_bidirectional", dest="encoder_bidirectional", default=True, action="store_true")
    parser.add_argument("--encoder_unidirectional", dest="encoder_bidirectional", default=False, action="store_false")

    # Other arguments
    parser.add_argument('--seed', type=int, default=42)

    flags = vars(parser.parse_args())

    # Some checks on the flags
    if flags["generate_vocabularies"]:
        assert flags["input_vocab_path"] and flags["target_vocab_path"], "Please specify paths to vocabularies to save."

    if flags["mode"] == "train":
        train(**flags)
    # TODO: put in correct mode"s


if __name__ == "__main__":
    main()
