# gscan_seq2seq

## Getting Started

Make a virtualenvironment that uses Python 3.7 or higher:

```virtualenv --python=/usr/bin/python3.7 <path/to/virtualenv>```

Activate the environment and install the requirements with a package manager:

```{ source <path/to/virtualenv>/bin/activate; python3.7 -m pip install -r requirements; }```

## Contents

Sequence to sequence models for Grounded SCAN.

To train a model on a grounded SCAN dataset with a simple situation representation (for more information LINK TO gSCAN / paper), run:

```python3.7 -m seq2seq --mode=train --data_directory=<path/to/folder/with/dataset.txt/> --output_directory=<path/where/models/will/be/saved> --attention_type=bahdanau --max_training_iterations=200000```

With the default settings, this will train a model that will reproduce the results from the paper. (TODO: link to paper)

## Important arguments

- max_decoding_steps: reflect max target length in data