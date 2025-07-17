# Modelizer

## Learning Program Behavioral Models from Synthesized Input-Output Pairs

This repository contains the implementation of the Modelizer framework that was presented in the paper ["Learning Program Behavioral Models from Synthesized Input-Output Pairs"](https://doi.org/10.1145/3748720) by Tural Mammadov, Dietrich Klakow, Alexander Koller, and Andreas Zeller.

The readme file and documentation for the framework are currently being updated.
Please periodically check back for updates.

### Required Executables
- Python _ver. 3.10_ or higher
- Pandoc _ver. 2.19.2_ (only for the replication of the experiments)

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
- `modelizer/llm.py`: this script contains the implementation of the LLM inference and fine-tuning using unsloth.ai framework
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
- `scripts/model-llm.py`: this script can be used to fine-tune LLMs
- `scripts/eval-llm.py`: this script can be used to evaluate
- `scripts/model-setup.py`: this script finds the hyperparameters for the models
- `scripts/model-train.py`: this script trains the models
- `scripts/model-tune.py`: this script performs the fine-tuning of trained models 
- `scripts/environmnet_setup.sh`: the helper bash script that sets up the environment for the experiments


### Replication of the Experiments
You can download the evaluation artifact from the following link: [https://zenodo.org/records/15041168](https://zenodo.org/records/15041168)

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
11) If you want to reproduce experiments with querying LLMs locally you can use the `scripts/eval-llm.py` script, or if you want to fine-tune the model using the LLM framework, you can use the `scripts/model-llm.py` script. The script will fine-tune the model using the unsloth.ai framework. The script requires the path to the model instance and the path to the synthetic data. The script will create a new instance of the model with the fine-tuned parameters. To run the experiments you need to download `eval_llm.zip` (evaluation results), `llm.zip` (training and test data for experiments with LLMs), and/or optionally `llm_fine_tuned_models.zip` (checkpoints with weights of already fine-tuned models) and unpack them into `datasets` directory. Attention these experiments require upgrading to the latest version of modelizer which will additionally install unsloth.ai framework dependency. Querying LLMs will require significantly more resources than the local model evaluation, in particular you need to have access to a GPU with at least 24 GB of video memory. 
12) Plots can be regenerated using the notebooks in the `plots.zip` file which should be also positioned in the `datasets` directory.  


### License
The framework is distributed under the GNU General Public License v3.0 or later. The license can be found in the `LICENSE` file.

### Citation
If you use the framework in your research, please cite the following paper:

```
@article{modelizer2025,
    author = {Mammadov, Tural and Klakow, Dietrich and Koller, Alexander and Zeller, Andreas},
    title = {Learning Program Behavioral Models from Synthesized Input-Output Pairs},
    year = {2025},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    issn = {1049-331X},
    url = {https://doi.org/10.1145/3748720},
    doi = {10.1145/3748720},
    note = {Just Accepted},
    journal = {ACM Trans. Softw. Eng. Methodol.},
    month = jul,
    keywords = {Software Testing, Mocking, Deep Learning}
}
```