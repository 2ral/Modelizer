from pathlib import Path
from datetime import date

if __name__ == "__main__":
    from sys import path as sys_path
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()

from modelizer.learner import Learner, load_model
from modelizer.utils import pickle_load, pickle_dump, parse_model_name, infer_subject, Multiprocessing


def __test_syn__(args: tuple) -> tuple[Path, list | None]:
    model_dir, src, trg, data_dir, simplified, load_tuned, disable_backend = args
    try:
        model = load_model(model_dir, disable_backend, load_tuned=load_tuned)
    except FileNotFoundError as e:
        print((str(e)), flush=True)
        return model_dir, None
    else:
        data = __load_data__(src, trg, data_dir, simplified)
        return model_dir, [__get_predictions__(model, data[src][i], data[trg][i]) for i in range(len(data[src]))]


def __test_real__(args: tuple) -> tuple[Path, list | None]:
    model_dir, src, trg, data_dir, simplified, load_tuned, disable_backend = args
    try:
        model = load_model(model_dir, disable_backend, load_tuned=load_tuned)
    except FileNotFoundError as e:
        print((str(e)), flush=True)
        return model_dir, None
    else:
        src_key = f"{src}_mapping"
        trg_key = f"{trg}_mapping"
        data = __load_data__(src, trg, data_dir, simplified)
        return model_dir, [__get_predictions__(model, data[src][i], data[trg][i], data[src_key][i], data[trg_key][i]) for i in range(len(data[src]))]


def __load_data__(src: str, trg: str, data_dir: Path, simplified: bool) -> dict[str, list]:
    assert src != trg, "Source and target must be different"
    match infer_subject(src, trg):
        case "sql":
            filepath = data_dir.joinpath("sql")
        case "markdown":
            filepath = data_dir.joinpath("markdown")
        case "expression":
            filepath = data_dir.joinpath("expression")
        case "mathml":
            filepath = data_dir.joinpath("mathml")
        case _:
            raise ValueError(f"Invalid source and target combination {src} - {trg}")
    data = pickle_load(filepath.joinpath("simplified.pickle" if simplified else "dataset.pickle"))
    assert data is not None, f"Data not found in {data_dir.name}"
    return data


def __get_predictions__(model: Learner, source_tokens, target_tokens, source_mapping: list | None = None, target_mapping: list | None = None):
    max_len = len(target_tokens)
    return {
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "source_mapping": source_mapping,
        "target_mapping": target_mapping,
        "b1":  model.translate(source_tokens, max_output_len=max_len),
        # "b2": model.translate_beamed(source_tokens, max_output_len=max_len, beam_size=2),
        # "b2top": model.translate_top_k(source_tokens, beam_size=2, max_output_len=max_len),
    }


def __parse_name__(model_path: Path):
    model_name = model_path.name
    ds, src, trg, part, simpl = parse_model_name(model_name.replace("ascii_math", "asciimath"))
    src = src.replace("asciimath", "ascii_math")
    trg = trg.replace("asciimath", "ascii_math")
    return model_path, ds, src, trg, part, simpl


def __get_configs__(parsed_values: list[tuple], data_dir: Path, load_tuned: bool = False, disable_backend: bool = False):
    return [(model_path, src, trg, data_dir, simpl, load_tuned, disable_backend) for model_path, ds, src, trg, _, simpl in parsed_values]


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--syn-dir', type=str, default="datasets/test/test_10k", help='Path to a Directory containing Tokenized Synthesized Test Data')
    arg_parser.add_argument('--real-dir', type=str, default="datasets/test/tokenized2", help='Path to a Directory containing Tokenized Real-World Test Data')
    arg_parser.add_argument('--model-dir', type=str, default="datasets/models", help='Path to a Directory containing Pre-Trained Models')
    arg_parser.add_argument('--simplified', action='store_true', help='Switch to Simplified Token Representation')
    arg_parser.add_argument('--model-count', type=int, default=4, help='Number of Models to Test in Parallel')
    arg_parser.add_argument('--cpu-only', action='store_true', help='Load model only to CPU')
    arg_parser.add_argument('--no-log', action='store_true', help='Disable TQDM Progress Bar Logging')
    parsed_args = arg_parser.parse_args()

    output_dir = Path("datasets/eval").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(parsed_args.model_dir).resolve()
    syn_test_dir = Path(parsed_args.syn_dir).resolve()
    real_test_dir = Path(parsed_args.real_dir).resolve()

    assert models_dir.is_dir(), "Models directory not found"
    assert syn_test_dir.is_dir(), "Synthesized test data directory not found"
    assert real_test_dir.is_dir(), "Real-world test data directory not found"
    directories = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("data_")
                   and d.name.endswith("simplified" if parsed_args.simplified else "enumerated")]

    parsed = [__parse_name__(m) for m in directories]
    results = {str(model_path): {
            "size": dataset_size,
            "source": source,
            "target": target,
            "partitioned": partitioned,
            "simplified_tokens": simplified_tokens,
        } for model_path, dataset_size, source, target, partitioned, simplified_tokens in parsed}

    test_results = Multiprocessing.parallel_run(__test_syn__, __get_configs__(parsed, syn_test_dir, False, parsed_args.cpu_only),
                                                None if parsed_args.no_log else "Testing models with Synthetic Data", parsed_args.model_count)
    for model_path, predictions in test_results:
        results[str(model_path)]["syn_results"] = predictions

    test_results = Multiprocessing.parallel_run(__test_real__, __get_configs__(parsed, real_test_dir, False, parsed_args.cpu_only),
                                                None if parsed_args.no_log else "Testing models with Real Data", parsed_args.model_count)
    for model_path, predictions in test_results:
        results[str(model_path)]["real_results"] = predictions
    test_results = Multiprocessing.parallel_run(__test_real__, __get_configs__(parsed, real_test_dir, True, parsed_args.cpu_only),
                                                None if parsed_args.no_log else "Testing fine-tuned models with Real Data", parsed_args.model_count)
    for model_path, predictions in test_results:
        results[str(model_path)]["tuned_results"] = predictions
    today = date.today().strftime("%d/%m/%Y").replace("/", "_")
    pickle_dump(list(results.values()), output_dir.joinpath(f"evaluation_results_{today}_{'simplified' if parsed_args.simplified else 'enumerated'}.pickle"))
