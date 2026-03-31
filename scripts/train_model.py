# This scripts trains a Modelizer instance using all supported model engines.
from pathlib import Path

from sys import path as sys_path

sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()

from modelizer import Trainer
from modelizer.tokenizers import SentencePieceTokenizer
import argparse


def run_training(trainer: Trainer):
    # Optional phase that enables program output formating
    # from modelizer.utils import DataHandlers
    # trainer.arguments.post_formating = DataHandlers.post_formating

    # Loading the dataset
    df_shuffled, train_data, test_data = trainer.load_dataset(False)

    # Training the tokenizer
    tokenizer, output_tokenizer = trainer.train_encoder_decoder_tokenizers(df_shuffled)

    # Training the model
    trainer.execute(
        config=None,  # Forge config from arguments inside execute
        train_data=train_data,
        test_data=test_data,
        tokenizer=tokenizer,
        output_tokenizer=output_tokenizer,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a Modelizer instance using all supported model engines.", add_help=False)
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message and exit, including Trainer options.')
    args, unknown = parser.parse_known_args()
    if args.help:
        parser.print_help()
        from modelizer.trainer import Trainer
        print("\n--- Trainer options ---\n")
        Trainer.print_help()
        return

    # Initialize the trainer
    # Either specify TrainingArguments manually:
    # - from modelizer import TrainingArguments
    # - arguments = TrainingArguments(...)
    # - trainer = Trainer(arguments=arguments)
    # or specify a path to a file containing the arguments:
    # - trainer = Trainer(arguments="path/to/config.pkl")
    # or initialize trainer using command line arguments (default):
    # - call Trainer.print_help() to see available options
    # - trainer = Trainer()
    model_trainer = Trainer()

    # Setting encoder-decoder tokenizers
    model_trainer.arguments.source_tokenizer_class = SentencePieceTokenizer
    model_trainer.arguments.target_tokenizer_class = SentencePieceTokenizer

    # Executing the training process
    run_training(model_trainer)


if __name__ == "__main__":
    main()
