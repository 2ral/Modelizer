# Modelizer
An open-source framework for training program-specific models from scratch or fine-tuning existing models using your own data.

#### Required Executables
 - Python _ver. 3.10_ or higher

### Installation
###### Installation from source:
1. Clone the repository:
2. Create and activate a virtual environment with Python 3.10 or higher:
3. Navigate to the cloned repository
4. Execute the installation command `pip install -e .`

### Common Installation Errors

#### ModuleNotFoundError: No module named 'torch'

Fix: run one of the commands depending on your Operating System type in the terminal and restart the installation:

- Windows: `pip3 install torch --index-url https://download.pytorch.org/whl/cu128`
- Linux and MacOS: `pip install torch`


#### /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found 

##### If you are using conda, run the following command in the terminal and restart the installation:
```
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
##### System-wide fix for Linux OS:
```
sudo apt update
sudo apt install libstdc++6
```

#### OSError: [Errno 2] No such file or directory on Windows OS
Fix: run the following command in the Elevated PowerShell terminal and restart the installation:

`New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`

#### clang++: error: unsupported option '-fopenmp' during installation of xformers
If this problem occurs on MacOS install llvm from HomeBrew:
```
brew install llvm
```

Then update environment variables:
```
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export CC="/opt/homebrew/opt/llvm/bin/clang"
export CXX="/opt/homebrew/opt/llvm/bin/clang++"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
```
Please use the correct HomeBrew path.


### Common Runtime Errors

#### NotImplementedError: The operator 'aten::_nested_tensor_from_mask_left_aligned' is not currently implemented for the MPS device.
This error occurs when using the MPS device on MacOS. There are two possible solutions: 
1. Use CPU device instead of MPS. To do this, set force_cpu=True in the config file. This option will significantly slow down the training process.
2. Specify the additional environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` before running the script. This option will slow down the training process, but it will allow you to use the MPS device. An example code snippet comes below.
```import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

#### Repository Content:
Here is the structure of the repository with main directories and files used in the project:

- /src/modelizer - main framework directory with all the code files
- /scripts - directory with scripts for training, testing and repairing models
- /pyproject.toml - file with package metadata and dependencies
- /Dockerfile - file with instructions to build the Docker container
- /LICENSE.txt - license file
- /README.md - this file

#### Helper commands to run training and repair:
To run the training start the container and execute the following command:
```
python scripts/train_model.py --subject <subject_name> --dataset <dataset_path> --source <source_name> --target <target_name> --trials 250 --train_epochs 100 --test_epochs 5 --test_size 10000 --batch-size 1 --fast
```
Attention! <source_name> and <target_name> should be the same as in the dataset.

To get help on all the available options for training and repair scripts, run the following command:
```
python scripts/train_model.py --help
```

To run the self-repair evaluation start the container and execute the following command:
```
python scripts/model_repair.py --subject <subject_name> --num-samples 0 --trials 25 --epochs 5 --root-dir <root_dir>
```
where <root_dir> is the directory with the trained model and the dataset, e.g. `artifacts/bc`. The `num-samples` option specifies the number of samples to be repaired. If it is set to 0, all samples from the test set will be repaired. The `trials` option specifies the number of repair trials for each sample, and the `epochs` option specifies the number of epochs for each repair trial.

To get help on all the available options for the repair script, run the following command:
```
python scripts/model_repair.py --help
```

#### Docker Container:
We provide a Docker container with all the dependencies pre-installed to simplify the installation process and avoid potential issues with dependency conflicts. 
The container is available on the Docker Hub repository which you can download using the following command:
```
docker pull turalmammadov/modelizer:latest
```

#### Running the container:
Run the container with the following command:
```
docker run --gpus all -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 turalmammadov/modelizer
```
