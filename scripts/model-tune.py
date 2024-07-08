from tqdm import tqdm
from pathlib import Path
from sys import path as sys_path
if __name__ == "__main__":
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()
from modelizer.learner import load_model
from modelizer.dataset import TorchDataset
from modelizer.utils import Logger, LoggingLevel, infer_subject, parse_model_name

if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of Epochs to Train')
    arg_parser.add_argument('--data-dir', type=str, default="datasets/test/tokenized", help='Path to a Directory containing Tokenized Real-World Test Data')
    arg_parser.add_argument('--model-dir', type=str, default="datasets/models", help='Path to a Directory containing Pre-Trained Models')
    arg_parser.add_argument('--simplified', action='store_true', help='Switch to Simplified Token Representation')
    args = arg_parser.parse_args()

    models_dir = Path(args.model_dir).resolve()
    data_dir = Path(args.data_dir).resolve()

    assert models_dir.is_dir(), "Models directory not found"
    assert data_dir.is_dir(), "Real-world test data directory not found"
    directories = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("data_")
                   and d.name.endswith("simplified" if args.simplified else "enumerated")]
    logger = Logger(LoggingLevel.INFO, "datasets/eval", file_logger=False)

    for m in tqdm(directories, desc="Fine-tuning models", total=len(directories)):
        dataset_size, source, target, partitioned, simplified_tokens = parse_model_name(m.name.replace("ascii_math", "asciimath"))
        source = source.replace("asciimath", "ascii_math")
        target = target.replace("asciimath", "ascii_math")
        model = load_model(m, logger=logger)
        assert source != target, "Source and target must be different"
        match infer_subject(source, target):
            case "sql":
                filepath = data_dir.joinpath("sql")
            case "markdown":
                filepath = data_dir.joinpath("markdown")
            case "expression":
                filepath = data_dir.joinpath("expression")
            case "mathml":
                filepath = data_dir.joinpath("mathml")
            case _:
                raise ValueError(f"Invalid source and target combination {source} - {target}")
        filepath = filepath.joinpath("simplified.pickle" if simplified_tokens else "dataset.pickle")
        dataset = TorchDataset(filepath, source, target, {source: model.src_vocabulary, target: model.trg_vocabulary})
        model.fine_tune(dataset.get_dataloader(), m, args.epochs)
