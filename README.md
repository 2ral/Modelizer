# Modelizer

## Learning Program Models using Generated Inputs

The readme file and documentation for the framework are currently being updated.
Please periodically check back for updates. We are sorry for the inconvenience, and thank you for your patience.

The missing link to the replication artifact will be added soon as well.
It was not included in the initial submission due to the size limitations of GitHub.


### Required Executables
- Python _ver. 3.10_ or higher
- Pandoc _ver. 2.19.2_

The subjects that were evaluated in the paper and their Python bindings require Python _ver. 3.10_. 
However, if you plan to use the framework with other subjects, you can use more recent versions of Python. 

### Installation
The framework is distributed as a Python module. 
To install it locally please repeat the following steps:
1) pull the repository
2) change your directory to the root of the repository
3) run the following command: ```python3 -m pip install -e .```

This will install the module in the editable mode, so you can modify the code and see the changes immediately.

#### Installation Errors

*ERROR: Could not build wheels for pygraphviz, which is required to install pyproject.toml-based projects*

The current Fuzzingbook Python module installation requires a PyGraphviz module, which further requires the installation of additional libraries.

If you encounter an error during installation, please follow the instructions listed in the [PyGrahviz documentation](https://pygraphviz.github.io/documentation/stable/install.html).

### Framework Structure
The framework is structured as follows:

`modelizer/` - the Python module that contains the implementation of the framework
- `modelizer/generators/`: this package contains the base class input generator as well as subject-specific implementations used in our experiments 
- `modelizer/subjects/`:  this package contains the bindings for the subjects that were evaluated in the paper
- `modelizer/tokenizer/`: this package contains the base implementation of the mapped-tokenization algorithm as well as subject-specific tokenizers
- `modelizer/dataset.py`: this script contains the implementation of the dataset class used to load and preprocess, as well as additional utility functions to form and process vocabularies
- `modelizer/learner.py`: the main script that implements the learning algorithm
- `modelizer/metrics.py`: this script contains the implementation of the metrics used to evaluate the models 
- `modelizer/optimizer.py`: this script provides the implementation hyperparameter optimization routine using the Optuna library 
- `modelizer/trainer.py`: an example implementation of initializing the model training or tuning
- `modelizer/transformer.py`: this script provides the implementation of the transformer model using the PyTorch library
- `modelizer/utils.py`: supplementary script to simplify data loading, handling, and logging

`scripts/` - the helper scripts that assist in running experiments. It can be modified to run experiments on different subjects.
- `scripts/compute-params.py`: this script retrieves the total number of trainable parameters in the model
- `scripts/compute-scores.py`: this script can help to compute the scores for the models. Attention! Significant computational resources can be required if executed on more than one core.
- `scripts/data-generate.py`: this script can be used to generate synthetic data for the subjects 
- `scripts/data-parse.py`: this script contains an example implementation for parsing and tokenizing new input-output pairs. The correct implementation for the tokenizers must be provided. 
- `scripts/model-eval.py`: this script can be used to evaluate the models
- `scripts/model-setup.py`: this script finds the hyperparameters for the models
- `scripts/model-train.py`: this script trains the models
- `scripts/model-tune.py`: this script performs the fine-tuning of trained models 
- `scripts/environmnet_setup.sh`: the helper bash script that sets up the environment for the experiments


### Replication Artifact
You can download the replication artifact from the following link:
COMING SOON. Please check back later.

The artifact is structured as follows:
- `train.zip`: synthetic data used for training
- `test.zip`: synthetic and extracted real data used for testing
- `eval.zip`: evaluation results (experiment results and notebooks with the analysis of the results)
- `models.zip`: all the models that were trained during the experiments
