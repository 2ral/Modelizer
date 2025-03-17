import argparse
from pathlib import Path
from sys import path as sys_path
if __name__ == "__main__":
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()
from modelizer.utils import infer_subject
from modelizer.llm import ModelType, FineTunedModel, INSTRUCTION_TEMPLATE

HF_KEY = "Put Your Huggingface API Key Here"
WANDB_KEY = "Put Your Wandb API Key Here"

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of Epochs to Train')
    arg_parser.add_argument('-s', '--source', type=str, help='Source type name')
    arg_parser.add_argument('-t', '--target', type=str, help='Target type name')
    arg_parser.add_argument('-m', '--model', type=str, help='Model type name')
    arg_parser.add_argument('--data', type=str, default="datasets/llm",
                            help='Path to a Directory containing training data')
    arg_parser.add_argument('--size', type=str, default="10k", help='Size of the dataset')
    arg_parser.add_argument('--batch', type=int, default=1, help='Batch size')
    arg_parser.add_argument('--output', type=str, default="modelizer_llm",
                            help='Path to a Directory to save the fine-tuned model')

    arg_parser.add_argument("--wandb", type=str, default=WANDB_KEY, help="Wandb API Key")
    arg_parser.add_argument("--hf", type=str, default=HF_KEY, help="Huggingface API Key")

    args = arg_parser.parse_args()
    source = args.source
    target = args.target
    subject = infer_subject(source.lower(), target.lower())
    model = ModelType(args.model)
    if model == ModelType.UNKNOWN:
        model = args.model
    epochs = args.epochs

    batch_size = args.batch
    assert isinstance(epochs, int), "Epochs must be an integer"
    assert epochs > 0, "Epochs must be greater than 0"
    assert isinstance(batch_size, int), "Batch size must be an integer"
    assert batch_size > 0, "Batch size must be greater than 0"

    data_dir = Path(args.data).resolve()
    assert data_dir.is_dir(), "Data directory not found"
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fine_tuned_model = FineTunedModel(model, source, target, output_dir,
                                      hf_token=HF_KEY, wandb_token=WANDB_KEY,
                                      model_instruction=INSTRUCTION_TEMPLATE)

    train_filepath = data_dir.joinpath("train").joinpath(args.size).joinpath(subject).joinpath("data.csv")
    test_filepath1 = data_dir.joinpath("test").joinpath(f"{subject}.json")
    test_filepath2 = data_dir.joinpath("test2").joinpath(f"{subject}.json")
    assert train_filepath.is_file(), f"Training data not found: {train_filepath}"
    assert test_filepath1.is_file(), f"Test data not found: {test_filepath1}"
    assert test_filepath2.is_file(), f"Test data not found: {test_filepath2}"
    fine_tuned_model.train(train_filepath, epochs, batch_size=batch_size)
    fine_tuned_model.test(test_filepath1, test_name="RawData")
    fine_tuned_model.test(test_filepath2, test_name="ProcessedData")
