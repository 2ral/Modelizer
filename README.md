# Modelizer

## Learning Program Models using Generated Inputs

The readme file and documentation for the framework are currently being updated.
Please periodically check back for updates.

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


### Replication of the Experiments
You can download the evaluation artifact from the following link: [https://zenodo.org/records/12742928](https://zenodo.org/records/12742928)

The artifact contains the data, models, and evaluation results for the experiments described in the paper and is structured as follows:
- `train.zip`: synthetic data used for the model training
- `test.zip`: synthetic and extracted real data used for the model evaluation
- `eval.zip`: evaluation results (experiment results and notebooks with the analysis of the results)
- `models.zip`: all the models that were trained during the experiments

The steps to replicate the experiments are as follows:
1) Clone the repository
2) Install the framework
3) Download artifact from the provided link
4) Create a new directory `datasets` next to the `modelizer` directory.
5) Unzip the downloaded archives into the `datasets` directory.
6) If you want to run the hyperparameter optimization, you can use the `scripts/model-setup.py` script. The script will find the optimal hyperparameters for the given input/output formats. For configuration please check the commandline arguments.  The new hyperparameters might be different from the ones used in the paper. Alternatively, you can reuse the hyperparameters that are included in the `train.zip` and move to the next step. 
7) Execute the model training for the given subject using `scripts/model-train.py` script. The script will train the model using the synthetic data. For configuration please check the commandline arguments.
8) Model can be fine-tuned using the `scripts/model-tune.py` script. It will not overwrite the original model but create a new instance of the model with the fine-tuned parameters.
9) `scripts/model-eval.py` script can be used to evaluate the trained model. The script will evaluate the model using the synthetic and real data and produce evaluation results as a `.pickle` file. Please specify the directory which contains the trained models instances. By default it is `datasets/models`. It can evaluate both pre-trained and fine-tuned models at the same time. 
10) The newly created `.pickle` file with evaluation results ca be found in the `datasets/eval` directory. There you can also find the notebooks which contain the analysis of the results our experiments. In the beginning of the every notebook the input file is specified. Please change the path to the file if you want to analyze the results of the new experiment.

_More detailed instructions are in preparation._

### Citation
If you use the framework in your research, please cite the following paper:

```
@misc{mammadov2024learningprogrambehavioralmodels,
      title={Learning Program Behavioral Models from Synthesized Input-Output Pairs}, 
      author={Tural Mammadov and Dietrich Klakow and Alexander Koller and Andreas Zeller},
      year={2024},
      eprint={2407.08597},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2407.08597}, 
}
```