#!/bin/sh

check_file() {
    if ! [ -f "$work_dir/$1" ]; then
        echo "The installation cannot continue!"
        echo "File '$1' does not exist in '$work_dir'." >&2
        exit 1
    fi
}

check_directory() {
    if ! [ -d "$work_dir/$1" ]; then
        echo "The installation cannot continue!"
        echo "Directory'$1' does not exist in '$work_dir'." >&2
        return 1
    fi
}

check_python_version() {
    local version=$("$1" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
    IFS='.' read -r -a version_array <<< "$version"

    if (( version_array[0] > 3 )) || { (( version_array[0] == 3 )) && (( version_array[1] >= 10 )); }; then
        echo "Python version is '$version'"
        return 0
    else
        echo "Python version is lower than 3.10: '$version'"
        return 1
    fi
}

check_modelizer_installation() {
    local module="modelizer"
    if pip3 show "$module" &> /dev/null; then
        echo "'$module' module is already installed."
    else
        echo "Installing '$module'"
        check_file "setup.py"
        check_file "pyproject.toml"
        check_directory "modelizer"
        pip3 install -e .
    fi
}



# Initialize working direcoty
if [ $# -eq 0 ]; then
    work_dir=$(pwd)
elif [ $# -eq 1 ]; then
    work_dir=$1
else
    echo "Error: This script accepts at most 1 argument"
    exit 1
fi

# Check if python3 is installed
if command -v python3 &> /dev/null; then
    if check_python_version python3; then
        if command -v pip3 &> /dev/null; then
            check_modelizer_installation``
        else
            echo "pip is not installed. Please install pip and try again."
            exit 1
        fi
    else
        echo "Please install the correct Python version or change your Python environment."
        exit 1
    fi
else
    echo "Python 3 is not installed. Please install Python 3.10 or"
    exit 1
fi
