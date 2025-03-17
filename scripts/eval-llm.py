import argparse
from pathlib import Path
from sys import path as sys_path
if __name__ == "__main__":
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()
from modelizer.utils import infer_subject
from modelizer.llm import Model, ModelType, INSTRUCTION_TEMPLATE

HF_KEY = "Put Your Huggingface API Key Here"

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-s', '--source', type=str, help='Source type name')
    arg_parser.add_argument('-t', '--target', type=str, help='Target type name')
    arg_parser.add_argument('-m', '--model', type=str, help='Model type name')
    arg_parser.add_argument('--data', type=str, default="datasets/llm",
                            help='Path to a Directory containing test data')
    arg_parser.add_argument('--output', type=str, default="modelizer_llm",
                            help='Path to a Directory to save evaluation results')

    args = arg_parser.parse_args()
    source = args.source
    target = args.target
    subject = infer_subject(source.lower(), target.lower())
    model = ModelType(args.model)
    if model == ModelType.UNKNOWN:
        model = args.model

    data_dir = Path(args.data).resolve()
    assert data_dir.is_dir(), "Data directory not found"
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Model(model, source, target, output_dir,
                  hf_token=HF_KEY, model_instruction=INSTRUCTION_TEMPLATE)
    model.init_model()
    test_filepath1 = data_dir.joinpath("test").joinpath(f"{subject}.json")
    test_filepath2 = data_dir.joinpath("test2").joinpath(f"{subject}.json")
    assert test_filepath1.is_file(), f"Test data not found: {test_filepath1}"
    assert test_filepath2.is_file(), f"Test data not found: {test_filepath2}"
    model.test(test_filepath1, test_name="RawData")
    model.test(test_filepath2, test_name="ProcessedData")



