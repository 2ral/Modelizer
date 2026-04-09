# this is a mock script that demonstrates how to use trained model for inference

if __name__ == "__main__":
    from pathlib import Path
    from sys import path as sys_path
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()

from modelizer import Modelizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Modelizer instance.", add_help=False)
    parser.add_argument('--path', '-p', type=str, required=True, help="Path to a directory with Modelizer checkpoint")
    parsed_args = parser.parse_args()
    # certainly it is inefficient to load model for every single inference iteration. write more efficient loop such that model is loaded once
    modelizer = Modelizer(parsed_args.path)
    test_input = None  # specify some real input
    # better execute this call in a loop
    prediction = modelizer.generate(input_data=test_input)
    return prediction
