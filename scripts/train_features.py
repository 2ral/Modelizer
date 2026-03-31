# This scripts trains a Modelizer instance that model program executions features to program inputs problems.
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train a Modelizer instance that models program execution features to program inputs.", add_help=False)
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message and exit, including Trainer options.')
    args, unknown = parser.parse_known_args()
    if args.help:
        parser.print_help()
        from modelizer.trainer import Trainer
        print("\n--- Trainer options ---\n")
        Trainer.print_help()
        return

    from pathlib import Path
    from datetime import datetime

    from sys import path as sys_path
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()

    from modelizer import Trainer
    from modelizer.utils import DataHandlers
    from modelizer.configs import SPACE_TOKEN
    from modelizer.tokenizers import SentencePieceTokenizer, FeatureTokenizer

    # Initialize the trainer
    trainer = Trainer()
    trainer.arguments.post_formating = DataHandlers.post_formating

    # Initialize Subject
    try:
        from modelizer.generators.implementations import init_subject
    except (ImportError, ModuleNotFoundError):
        from modelizer.generators.subjects import BaseSubject

        def init_subject(name: str) -> BaseSubject:
            raise NotImplementedError(f"implement here your own init_subject function to initialize {name}")

    trainer.arguments.subject_instance = init_subject(trainer.arguments.subject)

    # Loading the dataset
    df_shuffled, train_data, test_data = trainer.load_feature_dataset(False)

    # Training the tokenizer
    start_time = datetime.now()
    tokenizer = FeatureTokenizer(None)
    output_tokenizer = SentencePieceTokenizer(None)
    tokenizer.max_mutations = trainer.arguments.kwargs["max_features_mutations"]
    tokenizer.train(df_shuffled[trainer.arguments.source].tolist(),
                    encoding_policy=trainer.arguments.kwargs["feature_encoding"],
                    forging_policy=trainer.arguments.kwargs["feature_forging"],
                    legacy_padding_mode=trainer.arguments.legacy_padding_mode)
    output_tokenizer.train(
        data=df_shuffled[trainer.arguments.target].tolist(),
        separator=SPACE_TOKEN if trainer.arguments.kwargs["train_from_input_patterns"] else None,
        legacy_padding_mode=trainer.arguments.legacy_padding_mode
    )

    trainer.arguments.kwargs["tokenizer"] = tokenizer
    trainer.arguments.kwargs["output_tokenizer"] = output_tokenizer
    trainer.logger.info(f"Tokenizers trained in {trainer.get_time_diff(start_time)} | {tokenizer.__class__.__name__} | {output_tokenizer.__class__.__name__}")

    # Training the model
    trainer.execute(
        config=None,  # Forge config from arguments inside execute
        train_data=train_data,
        test_data=test_data,
        tokenizer=tokenizer,
        output_tokenizer=output_tokenizer,
    )

if __name__ == "__main__":
    main()
