# This script uses feedback loop to repair trained Modelizer models.

from pathlib import Path
from datetime import datetime
from pandas import read_csv

from sys import path as sys_path

sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()

from modelizer.repairer import Repairer
from modelizer.utils import Pickle, DataHandlers


def run_repair(repairer: Repairer):
    # Loading the dataset
    repairer.logger.info("Loading dataset...")
    data = []
    datasets = repairer.arguments.get_dataset_paths()
    files_string = '\n'.join([d.as_posix() for d in datasets])
    repairer.logger.info(f"Loading dataset from files:{files_string}")
    for dataset_path in datasets:
        if dataset_path.suffix.lower() == ".csv":
            # noinspection PyArgumentList
            loaded = read_csv(dataset_path.as_posix())
            data.extend(list(zip(loaded['Input'], loaded['Expected'], loaded['Predicted'])))
        else:
            loaded = Pickle.load(dataset_path)
            data.extend([(d['Input'], d['Expected'], d['Predicted']) for d in loaded])
    repairer.logger.info(f"Dataset loaded. Total Records: {len(data)}")

    # Preprocessing data and repairing
    processed_data = repairer.preprocess_data(data)
    repairer.repair(processed_data)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Repair trained Modelizer models.", add_help=False)
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message and exit, including Repairer options.')
    args, unknown = parser.parse_known_args()
    if args.help:
        parser.print_help()
        print("\n--- Repairer options ---\n")
        Repairer.print_help()
        return

    init_time = start_time = datetime.now()
    # Parsing command line arguments
    arguments = Repairer.parse_arguments()

    # Initializing the subject instance

    try:
        from modelizer.generators.implementations import init_subject
    except (ImportError, ModuleNotFoundError):
        from modelizer.generators.subjects import BaseSubject

        def init_subject(name: str) -> BaseSubject:
            raise NotImplementedError(f"implement here your own init_subject function to initialize {name}")

    arguments.subject_instance = init_subject(arguments.subject)

    # Initializing the Repairer
    # Either specify RepairArguments manually:
    # - from modelizer import RepairArguments
    # - arguments = RepairArguments(...)
    # - repairer = Repairer(arguments)
    # or specify a path to a file containing the arguments:
    # - repairer = Repairer(arguments="path/to/config.pkl")
    # or initialize repairer using command line arguments (default):
    # - call Repairer.print_help() to see available options
    # - arguments = Repairer.parse_arguments()
    # - repairer = Repairer(arguments)
    model_repairer = Repairer(arguments)

    # Optionally configuring the validator's post formating function
    model_repairer.validator.config.post_formating_func = DataHandlers.post_formating

    # Executing the repair process
    run_repair(model_repairer)


if __name__ == "__main__":
    main()
