from seq2seq.gSCAN_dataset import GroundedScanDataset

import argparse
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger(__name__)


def main():

    parser = argparse.ArgumentParser(description="Sequence to sequence models for Grounded SCAN")
    parser.add_argument('--mode', type=str, default='test', help='train, test or predict')
    parser.add_argument('--data_directory', type=str, default='data', help='Path to folder with data.')
    parser.add_argument('--data_path', type=str, default='data/dataset.txt', help='Path to file with data.')
    parser.add_argument('--input_vocab_path', type=str, default='data/training_input_vocab.txt',
                        help='Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py')
    parser.add_argument('--target_vocab_path', type=str, default='data/training_target_vocab.txt',
                        help='Path to file with target vocabulary as saved by Vocabulary class in gSCAN_dataset.py')
    parser.add_argument('--save_vocabularies', dest='save_vocabularies', default=False, action='store_true')

    flags = vars(parser.parse_args())

    # Some checks on the flags
    if flags['save_vocabularies']:
        assert flags['input_vocab_path'] and flags['target_vocab_path'], "Please specify paths to vocabularies to save."

    # TODO: put in correct mode's
    training_set = GroundedScanDataset(flags["data_path"], flags["data_directory"], split="train",
                                       input_vocabulary_file=flags['input_vocab_path'],
                                       target_vocabulary_file=flags['target_vocab_path'])
    logger.info("Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    if flags['save_vocabularies']:
        training_set.save_vocabularies(flags["input_vocab_path"], flags["target_vocab_path"])
    test_set = GroundedScanDataset(flags["data_path"], flags["data_directory"], split="test",
                                   input_vocabulary_file=flags["input_vocab_path"],
                                   target_vocabulary_file=flags["target_vocab_path"])
    logger.info("Input vocabulary size test set: {}".format(test_set.input_vocabulary_size))
    logger.info("Output vocabulary size test set: {}".format(test_set.target_vocabulary_size))


if __name__ == "__main__":
    main()
