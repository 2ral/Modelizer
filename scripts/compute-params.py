from pathlib import Path
from math import floor as math_floor

from tqdm.auto import tqdm
from pandas import DataFrame


if __name__ == "__main__":
    from sys import path as sys_path
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()
from modelizer.learner import load_model
from modelizer.utils import parse_model_name


def __parse_name__(model_path: Path):
    model_name = model_path.name
    ds, src, trg, part, simpl = parse_model_name(model_name.replace("ascii_math", "asciimath"))
    src = src.replace("asciimath", "ascii_math")
    trg = trg.replace("asciimath", "ascii_math")
    return model_path, ds, src, trg, part, simpl


def __count_parameters__(arguments: tuple) -> dict[str, str | int]:
    model_path, ds, src, trg, part, simpl = arguments
    model = load_model(model_path, disable_backend=False)
    result = {
        "Dataset": ds,
        "Source": src,
        "Target": trg,
        "Parameters": model.count_parameters(),
        "Partitioned": part,
        "Simplified": simpl,
    }
    return result


if __name__ == "__main__":
    models_dir = Path("datasets/models").resolve()
    output_dir = Path("datasets/eval").resolve()
    parsed = [__parse_name__(m) for m in [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("data_")]]
    results = [__count_parameters__(args) for args in tqdm(parsed, desc="Counting Parameters...")]
    params = math_floor(sum([entry["Parameters"] for entry in results]) / len(results))
    df = DataFrame(results)
    df.to_excel(output_dir.joinpath("model_parameters.xlsx"), index=False)
    print("Average Parameters:", params)
