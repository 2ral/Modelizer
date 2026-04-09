# This scripts trains a Modelizer instance using all supported model engines.
import argparse

if __name__ == "__main__":
    from pathlib import Path
    from sys import path as sys_path
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()

from modelizer import Trainer


def run_training(trainer: Trainer):
    # Optional phase that enables program output formating
    # from modelizer.utils import DataHandlers
    # trainer.arguments.post_formating = DataHandlers.post_formating

    # Loading the dataset
    df_shuffled, train_data, test_data = trainer.load_dataset(False)

    # Training the tokenizer
    tokenizer, output_tokenizer = trainer.train_seq2seq_tokenizers(df_shuffled)

    # Training the model
    trainer.execute(
        config=None,  # Forge config from arguments inside execute
        train_data=train_data,
        test_data=test_data,
        tokenizer=tokenizer,
        output_tokenizer=output_tokenizer,
    )


def main():
    from modelizer import Trainer, SentencePieceTokenizer

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
    trainer = Trainer()

    # Setting encoder-decoder tokenizers
    trainer.arguments.source_tokenizer_class = SentencePieceTokenizer
    trainer.arguments.target_tokenizer_class = SentencePieceTokenizer

    # Executing the training process
    run_training(trainer)


if __name__ == "__main__":
    main()
