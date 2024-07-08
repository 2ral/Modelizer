import argparse
from pathlib import Path
from tqdm.auto import tqdm
from pandas import DataFrame
from sys import path as sys_path
from gc import collect as gc_collect
if __name__ == "__main__":
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()
from modelizer.utils import pickle_load, pickle_dump, Multiprocessing
from modelizer.metrics import compute_score, compute_error_distribution


def process_file(filepath: Path, workers: int = 1):
    assert file.is_file(), f"File {file.as_posix()} not found"
    print(f"Loading {filepath.stem}...")
    data = pickle_load(filepath)
    assert isinstance(data, list)
    print("Data loaded.")
    if workers == 1:
        results = [compute_score(entry) for entry in tqdm(data, desc=f"Computing scores for {filepath.stem}...")]
    else:
        results = Multiprocessing.parallel_run(compute_score, data, text=f"Computing scores for {filepath.stem}...", n_jobs=workers)
    combined_scores = {}
    for entry in results:
        for score in entry:
            for k, v in score.items():
                if k in combined_scores:
                    combined_scores[k].append(v)
                else:
                    combined_scores[k] = [v]
    try:
        df = DataFrame(combined_scores, columns=list(combined_scores.keys()))
        print("Saving scores to a SpreadSheet...")
        df.to_excel(f"{root_dir.as_posix()}/{filepath.stem}.xlsx", index=False)
    except Exception as e:
        if pickle_dump(results, f"{root_dir.as_posix()}/{filepath.stem}_scores.pickle") == 0:
            print(f"Cannot save results as SpreadSheet so scores saved to {root_dir.as_posix()}/{filepath.stem}_scores.pickle")
        else:
            raise e
    finally:
        del df, results, combined_scores
        gc_collect()

    if workers == 1:
        results = [compute_error_distribution((entry, "real_results")) for entry in tqdm(data, desc=f"Computing error distribution of Real Evaluation for {filepath.stem}...")]
        results.extend([compute_error_distribution((entry, "tuned_results")) for entry in tqdm(data, desc=f"Computing error distribution of Tuned Evaluation for {filepath.stem}...")])
    else:
        arguments = [[entry, "real_results"] for entry in data]
        results = Multiprocessing.parallel_run(compute_error_distribution, arguments, text=f"Computing error distribution of Real Evaluation for {filepath.stem}...", n_jobs=workers)
        for arg in arguments:
            arg[1] = "tuned_results"
        results.extend(Multiprocessing.parallel_run(compute_error_distribution, arguments, text=f"Computing error distribution of Tuned Evaluation for {filepath.stem}...", n_jobs=workers))
    try:
        df = DataFrame(results)
        print("Saving error distribution to a SpreadSheet...")
        df.to_excel(f"{root_dir.as_posix()}/{filepath.stem.replace('evaluation_results', 'error_distribution')}.xlsx", index=False)
    except Exception as e:
        if pickle_dump(results, f"{root_dir.as_posix()}/{filepath.stem}_error_distribution.pickle") == 0:
            print(f"Cannot save error distribution as SpreadSheet so error distribution saved to {root_dir.as_posix()}/{filepath.stem.replace('evaluation_results', 'error_distribution')}.pickle")
        else:
            raise e
    finally:
        print(f"{filepath.stem} processing completed.")
        del df, results, data
        gc_collect()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-dir', type=str, default="datasets/eval", help='Root Directory for Evaluation Data')
    arg_parser.add_argument('--filename', type=str, default="", help="Optional Filename for Evaluation Data")
    arg_parser.add_argument('--workers', type=int, default=1, help='Optional Number of Workers to use for Parallel Processing')
    args = arg_parser.parse_args()
    assert len(args.data_dir) > 0, "Data directory not provided"
    root_dir = Path(args.data_dir).resolve()

    if len(args.filename):
        process_file(root_dir.joinpath(args.filename), args.workers)
    else:
        for file in root_dir.glob("evaluation_results_*.pickle"):
            process_file(file, args.workers)
