# This scripts trains a Modelizer instance using Transformer Encoder-Decoder models.
import argparse

if __name__ == "__main__":
    from pathlib import Path
    from sys import path as sys_path
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()


def main():
    from modelizer import Trainer, SentencePieceTokenizer

    parser = argparse.ArgumentParser(description="Train a Modelizer instance using Transformer Encoder-Decoder models.", add_help=False)
    parser.add_argument("--help", "-h", action="store_true", help="Show this help message and exit, including Trainer options.")
    args, unknown = parser.parse_known_args()
    if args.help:
        parser.print_help()
        from modelizer.trainer import Trainer
        print("\n--- Trainer options ---\n")
        Trainer.print_help()
        return

    # Initialize the trainer
    trainer = Trainer()

    # Optional phase that enables program output formating
    # from modelizer.utils import DataHandlers
    # trainer.arguments.post_formating = DataHandlers.post_formating

    # Loading the dataset
    df_shuffled, train_data, test_data = trainer.load_dataset(False)

    # Training the tokenizer
    tokenizer, output_tokenizer = trainer.train_seq2seq_tokenizers(df_shuffled, SentencePieceTokenizer, SentencePieceTokenizer)

    # Training the model
    # Change config="v1" to train a model using the Modelizer v1 configuration from:
    # "Learning Program Behavioral Models from Synthesized Input-Output Pairs." https://doi.org/10.1145/3748720
    # Change config="advanced" to train high-performance sequence-to-sequence transformer architecture
    # optimized for accuracy, memory efficiency and the ability to process long sequences.
    trainer.execute(
        config="seq2seq",
        train_data=train_data,
        test_data=test_data,
        tokenizer=tokenizer,
        output_tokenizer=output_tokenizer,
    )


if __name__ == "__main__":
    main()
