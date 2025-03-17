from math import isclose
from pathlib import Path
from datetime import datetime
from modelizer.learner import Learner, load_model
from multiprocessing import Process, Value, Array, set_start_method
from modelizer.dataset import TrainDataset, TorchDataset, load_vocabs, save_vocabs
from modelizer.utils import FileLogger, LoggingLevel, infer_subject, pickle_load, parse_model_name, chunkify_list


def get_training_args():
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of Epochs to Train')
    arg_parser.add_argument('-s', '--source', type=str, default="html", help='Source Data Type')
    arg_parser.add_argument('-t', '--target', type=str, default="markdown", help='Target Data Type')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='Show Debug Reports')
    arg_parser.add_argument('-c', '--clear', action='store_true', help='Clear Caches after every Epoch')
    arg_parser.add_argument('-v', '--vocab', action='store_true', help='Use Pre-computed Vocabularies')
    arg_parser.add_argument('-p', '--params', action='store_true', help='Load Hyperparameters from File')
    arg_parser.add_argument('--simplified', action='store_true', help='Switch to Simplified Token Representation')
    arg_parser.add_argument('--cpu', action='store_true', help='Run using CPU')
    arg_parser.add_argument('--train', type=str, default="data", help='Path to Directory containing Train Data')
    arg_parser.add_argument('--test', type=str, default="", help='Optional path to Directory containing Test Data')
    arg_parser.add_argument('--num-train-samples', type=int, default=0, help='Specify the Number of Train Samples to Use')
    arg_parser.add_argument('--num-test-samples', type=int, default=0, help='Specify the Number of Test Samples to Use')
    arg_parser.add_argument('--batch-size', type=int, default=1, help='Batch Size')
    arg_parser.add_argument('--pos-encoding-size', type=int, default=5500, help='Positional Encoding Size')
    arg_parser.add_argument('--train-fraction', type=float, default=0.8, help='Dataset Train Split Fraction')
    arg_parser.add_argument('--test-fraction', type=float, default=0.0, help='Dataset Test Split Fraction')
    arg_parser.add_argument('--shuffle-dataset', action='store_true', help='Shuffle Dataset before Splitting')
    # VALID FRACTION = DATALOADER_LEN * (1 - (1 - TEST FRACTION)  * TRAIN_FRACTION) if --test is not specified else DATALOADER_LEN * (1 - TRAIN_FRACTION)
    return arg_parser.parse_args()


def train_models(args, train_path: Path):
    set_start_method('spawn')
    process1 = TrainingProcess(args.source, args.target, args.simplified, args.epochs,
                               train_path.as_posix(), args.test,
                               args.num_train_samples, args.num_test_samples,
                               args.train_fraction, args.test_fraction, args.batch_size,
                               args.shuffle_dataset, args.pos_encoding_size, args.cpu_only)
    process2 = TrainingProcess(args.target, args.source, args.simplified, args.epochs,
                               train_path.as_posix(), args.test,
                               args.num_train_samples, args.num_test_samples,
                               args.train_fraction, args.test_fraction, args.batch_size,
                               args.shuffle_dataset, args.pos_encoding_size, args.cpu_only)
    process1.start()
    process2.start()
    process1.join()
    process2.join()


def tune_models(parsed_args):
    set_start_method('spawn')
    processes = []
    models_dir = Path(parsed_args.model_dir).resolve()
    assert models_dir.is_dir(), "Models directory not found"
    directories = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("data_")]
    for d in chunkify_list(list(directories), parsed_args.models):
        process = TuningProcess(parsed_args, d)
        process.start()
        processes.append(process)
    for p in processes:
        p.join()


class TrainingProcess(Process):
    def __init__(self, source: str, target: str,
                 simplified: bool, epochs: int,
                 train_dir: str, test_dir: str,
                 num_train_samples: int | float = 0, num_test_samples: int | float = 0,
                 train_fraction: float = 0.8, test_fraction: float = 0.0,
                 batch_size: int = 1, shuffle_dataset: bool = False,
                 pos_encoding_size: int = 5500, cpu_only: bool = False,
                 output_dir: str | Path = "/tmp/modelizer"):
        Process.__init__(self)
        self.output_dir = Path(output_dir).resolve() if isinstance(output_dir, str) else output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.source = source.lower()
        self.target = target.lower()
        self.simplified = simplified
        self.epochs = epochs
        self.train_data_dir = Path(train_dir).resolve()
        self.test_data_dir = Path(test_dir).resolve() if len(test_dir) else None
        self.train_samples = num_train_samples
        self.test_samples = num_test_samples
        self.train_fraction = train_fraction
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.pos_encoding_size = pos_encoding_size
        self.disable_backend = cpu_only

        # Hardcoded Parameters
        self.test_loss = Value('d', 0.0)
        self.train_losses = Array('d', [0.0 for _ in range(self.epochs)])
        self.valid_losses = Array('d', [0.0 for _ in range(self.epochs)])

    def run(self):
        assert self.train_data_dir.exists(), f"Data Directory {self.train_data_dir.as_posix()} does not exist"
        start_time = datetime.now()
        params_dir = self.train_data_dir.parent.joinpath("hyperparameters")
        root_dir = self.train_data_dir.joinpath("models").joinpath(f"modelizer_{self.source}_{self.target}")
        root_dir.mkdir(parents=True, exist_ok=True)
        logger = FileLogger(f"{self.train_data_dir.name}_{'simplified' if self.simplified else 'enumerated'}_{self.source}_{self.target}", LoggingLevel.INFO, root_dir.as_posix(), "debug.log")
        logger.info(f"Task: Model Training | Date and Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        subject = infer_subject(self.source, self.target)
        dataset_filename_filter = "simplified.*" if self.simplified else "dataset.*"

        vocabs_dir = self.train_data_dir.parent.joinpath("vocabs").joinpath(
            f"{subject}_simplified" if self.simplified else subject)
        vocabularies = load_vocabs(vocabs_dir)
        if vocabularies is not None:
            logger.info(f"Vocabularies loaded from {vocabs_dir.as_posix()}")

        logger.info("Loading Dataset...")
        train_data_dir = self.train_data_dir.joinpath(subject)
        assert train_data_dir.exists(), f"Train Data Directory {train_data_dir.as_posix()} does not exist"
        assert not train_data_dir.is_file(), f"Train Data Directory path {train_data_dir.as_posix()} points to a file"
        logger.info(f"Train Data Directory: {train_data_dir.as_posix()}")
        train_dataset = TrainDataset(list(train_data_dir.glob(dataset_filename_filter))[0], self.source, self.target, vocabularies, self.train_samples)
        logger.info(f"Dataset loaded in {datetime.now() - start_time} | Records: {len(train_dataset)}")

        if self.test_data_dir is not None:
            test_loading_start = datetime.now()
            if subject not in self.test_data_dir.as_posix():
                self.test_data_dir = self.test_data_dir.joinpath(subject)
            assert self.test_data_dir.exists(), f"Test Data Directory {self.test_data_dir.as_posix()} does not exist"
            assert not self.test_data_dir.is_file(), f"Test Data Directory path {self.test_data_dir.as_posix()} points to a file"
            logger.info(f"Test Data Directory: {self.test_data_dir.as_posix()}")

            test_dataset = TorchDataset(list(self.test_data_dir.glob(dataset_filename_filter))[0], self.source, self.target, train_dataset.vocabs, self.test_samples)
            test_data_loader = test_dataset.get_dataloader(batch_size=self.batch_size, shuffle=self.shuffle_dataset)
            logger.info(f"Dataset loaded in {datetime.now() - test_loading_start} | Records: {len(test_data_loader)}")
            train_d, eval_d, _ = train_dataset.get_dataloaders(self.train_fraction, self.test_fraction, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
            dataloaders = (train_d, eval_d, test_data_loader)
        else:
            if isclose(self.test_fraction, 0.0, abs_tol=0.00001):
                self.test_fraction = 0.2
            dataloaders = train_dataset.get_dataloaders(self.train_fraction, self.test_fraction, batch_size=self.batch_size, shuffle=self.shuffle_dataset)

        params = Learner.HYPERPARAMETERS.copy()
        param_file_candidates = [
            params_dir.joinpath(f"hyperparameters_learning_{self.source}_{self.target}_{'simplified' if self.simplified else 'enumerated'}.pickle"),
            params_dir.joinpath(f"hyperparameters_learning_{self.target}_{self.source}_{'simplified' if self.simplified else 'enumerated'}.pickle"),
            params_dir.joinpath(f"hyperparameters_model_{self.source}_{self.target}_{'simplified' if self.simplified else 'enumerated'}.pickle"),
            params_dir.joinpath(f"hyperparameters_model_{self.target}_{self.source}_{'simplified' if self.simplified else 'enumerated'}.pickle")
        ]

        for filepath in param_file_candidates:
            loaded = pickle_load(filepath)
            if loaded is not None:
                logger.info(f"Loaded Hyperparameter Configuration: {filepath.as_posix()}")
                params.update(loaded)
                break

        params["epoch"] = 0
        params["source"] = self.source
        params["target"] = self.target
        params["debug"] = False
        params["disable_backend"] = False
        params["free_cached_memory"] = False
        params["pos_encoding_size"] = self.pos_encoding_size

        logger.info(f"Initializing the Model...")
        learner = Learner(dataloaders=dataloaders, vocabularies=train_dataset.get_vocabularies(), params=params, logger=logger)
        save_vocabs(train_dataset.vocabs, root_dir)
        logger.info(f"Transformer model initialized in {datetime.now() - start_time}")
        logger.info(f"Training the Model for {self.epochs} epochs")
        result = learner.train(self.epochs, root_dir.joinpath("{}.{}").as_posix())
        logger.info(f"Model trained in {datetime.now() - start_time}" if result is not None else "Model training failed")
        if result is not None:
            self.test_loss.value = result["test_loss"]
            for i in range(self.epochs):
                self.train_losses[i] = result["train_loss"][i]
                self.valid_losses[i] = result["valid_loss"][i]
            root_dir.rename(self.output_dir.joinpath(f"{train_data_dir.parent.name}___{root_dir.name}_____{'simplified' if self.simplified else 'enumerated'}"))


class TuningProcess(Process):
    def __init__(self, parsed_args, directories: None | list[Path] = None):
        Process.__init__(self)
        self.epochs = parsed_args.epochs
        self.data_dir = Path(parsed_args.data_dir).resolve()
        self.output_dir = Path(parsed_args.output_dir).resolve().joinpath("models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        assert self.data_dir.is_dir(), "Real-world test data directory not found"
        if directories is not None:
            self.directories = directories
        else:
            models_dir = Path(parsed_args.model_dir).resolve()
            assert models_dir.is_dir(), "Models directory not found"
            self.directories = list(models_dir.iterdir())

    def run(self):
        logger = FileLogger("tuning", LoggingLevel.INFO, "datasets/eval")
        for m in self.directories:
            dataset_size, source, target, partitioned, simplified_tokens = parse_model_name(
                m.name.replace("ascii_math", "asciimath"))
            source = source.replace("asciimath", "ascii_math")
            target = target.replace("asciimath", "ascii_math")
            model = load_model(m, logger=logger)
            assert source != target, "Source and target must be different"
            match infer_subject(source, target):
                case "sql":
                    filepath = self.data_dir.joinpath("sql")
                case "markdown":
                    filepath = self.data_dir.joinpath("markdown")
                case "expression":
                    filepath = self.data_dir.joinpath("expression")
                case "mathml":
                    filepath = self.data_dir.joinpath("mathml")
                case _:
                    raise ValueError(f"Invalid source and target combination {source} - {target}")
            filepath = filepath.joinpath("simplified.pickle" if simplified_tokens else "dataset.pickle")
            dataset = TorchDataset(filepath, source, target, {source: model.src_vocabulary, target: model.trg_vocabulary})
            model.fine_tune(dataset.get_dataloader(), m, self.epochs)
            m.rename(self.output_dir.joinpath(m.name))
