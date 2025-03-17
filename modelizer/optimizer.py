import optuna
from pathlib import Path
from joblib import parallel_backend
from modelizer.learner import Learner
from modelizer.dataset import TrainDataset, TorchDataset, load_vocabs
from modelizer.utils import infer_subject, pickle_dump, pickle_load, Logger, FileLogger, LoggingLevel


class ParameterSearchOptimizer:
    def __init__(self, source: str, target: str, train_dir: str, test_dir: str):
        assert source != target, "Source and target must be different"
        self.logger = None
        self.source = source
        self.target = target
        self.params = Learner.HYPERPARAMETERS.copy()
        self.params["source"] = source
        self.params["target"] = target
        self.train_dir = Path(train_dir).resolve()
        self.test_dir = Path(test_dir).resolve()
        self.test_dir_extra = Path("datasets/test/tokenized").resolve()
        self.params_dir = Path("datasets/train/hyperparameters").resolve()
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.param_history = set()

        # Default values get overwritten by run_model_search() or run_learning_rate_search()
        self.name = None
        self.trials = 0
        self.start_up = 0
        self.warm_up = 0
        self.epochs = 1
        self.process_count = 1
        self.num_train_samples = 0
        self.num_test_samples = 0
        self.debug_mode = False
        self.simplified_tokens = False
        self.use_extra_test_samples = False
        self.load_vocabularies = False

        # Parameter Configuration
        self.config = {
            "feedforward_size": [1024, 2048, 4096, 8192],
            "embedding_size": [256, 512, 1024, 2048],
            "head_count": [8, 16, 32, 64],
            "num_encoder_layers": [1, 2, 3, 4, 5],
            "num_decoder_layers": [1, 2, 3, 4, 5],
            "clip_gradients": [None, 1.0],
            "dropout": [0.0, 0.1],
            "learning_policy": [None, "linear", "lambda", "multiplicative", "cosine", "step", "exponential"],
            "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01],
            "weight_decay": [0.0001, 0.0005, 0.001, 0.005, 0.01],
        }

    def __objective_model__(self, trial) -> float:
        assert not any([v is None for v in [self.source, self.target, self.train_dir, self.test_dir, self.logger]]), "Do not call this method directly"
        params = self.params.copy()
        params["feedforward_size"] = trial.suggest_categorical("feedforward_size", self.config["feedforward_size"])
        params["embedding_size"] = trial.suggest_categorical("embedding_size", self.config["embedding_size"])
        params["head_count"] = trial.suggest_categorical("head_count", self.config["head_count"])
        params["num_encoder_layers"] = trial.suggest_categorical("num_encoder_layers", self.config["num_encoder_layers"])
        params["num_decoder_layers"] = trial.suggest_categorical("num_decoder_layers", self.config["num_decoder_layers"])
        params["clip_gradients"] = trial.suggest_categorical("clip_gradients", self.config["clip_gradients"])
        params["dropout"] = trial.suggest_categorical("dropout", self.config["dropout"])
        return self.__train__(trial, params)

    def __objective_learning__(self, trial) -> float:
        assert not any([v is None for v in [self.source, self.target, self.train_dir, self.test_dir, self.logger]]), "Do not call this method directly"
        params = self.params.copy()
        params["learning_policy"] = trial.suggest_categorical("learning_policy", self.config["learning_policy"])
        params["learning_rate"] = trial.suggest_categorical("learning_rate", self.config["learning_rate"])
        params["weight_decay"] = trial.suggest_categorical("weight_decay", self.config["weight_decay"])
        return self.__train__(trial, params)

    def __train__(self, trial, params) -> float:
        assert not any([v is None for v in [self.source, self.target, self.train_dir, self.test_dir, self.logger]]), "Do not call this method directly"
        val_string = "".join([str(v) for v in params.values()])
        if val_string in self.param_history:
            raise optuna.TrialPruned()
        self.param_history.add(val_string)
        best_valid_loss = float("inf")
        model_state_dict = None
        subject = infer_subject(self.source, self.target)
        vocabs_dir = Path("datasets/train/vocabs").resolve().joinpath(f"{subject}_simplified" if self.simplified_tokens else subject)
        train_filepath = self.train_dir.joinpath(subject).joinpath("simplified.pickle" if self.simplified_tokens else "dataset.pickle")
        test_filepaths = [self.test_dir.joinpath(subject).joinpath("simplified.pickle" if self.simplified_tokens else "dataset.pickle")]
        if self.use_extra_test_samples and isinstance(self.test_dir_extra, Path) and self.test_dir_extra.is_dir():
            test_filepaths.append(self.test_dir_extra.joinpath(subject).joinpath("simplified.pickle" if self.simplified_tokens else "dataset.pickle"))
        vocabularies = load_vocabs(vocabs_dir) if self.load_vocabularies else None
        train_set = TrainDataset(train_filepath, self.source, self.target, vocabularies, self.num_train_samples)
        if all([f.is_file() for f in test_filepaths]):
            test_set = TorchDataset(tuple(test_filepaths), self.source, self.target, train_set.vocabs, self.num_test_samples)
            train_d, eval_d, _ = train_set.get_dataloaders(0.8, 0.0)
            dataloaders = (train_d, eval_d, test_set.get_dataloader())
        else:
            dataloaders = train_set.get_dataloaders(0.8, 0.2)
        learner = Learner(dataloaders, train_set.get_vocabularies(), params, logger=self.logger)
        scheduler = learner.initialize_scheduler()
        self.params["source_vocab_size"] = learner.params["source_vocab_size"]
        self.params["target_vocab_size"] = learner.params["target_vocab_size"]
        if self.debug_mode:
            self.logger.info(f"Trial {trial.number} | Params: {trial.params}")
            self.logger.info(f"Trial {trial.number} | The initialized Learner has {learner.count_parameters():,} trainable parameters.")
        for i in range(self.epochs):
            learner.train_epoch(learner.train_iter)
            if scheduler is not None:
                scheduler.step()
            valid_loss = learner.valid_epoch(learner.valid_iter)
            trial.report(valid_loss, i)
            if trial.should_prune():
                if self.debug_mode:
                    self.logger.info(f"Trial {trial.number} is getting pruned at Epoch {i} due to Valid-Loss = {valid_loss}")
                raise optuna.TrialPruned()
            elif self.debug_mode:
                self.logger.info(f"Trial {trial.number} | Epoch {i} completed with Valid-Loss = {valid_loss}")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model_state_dict = learner.model.state_dict().copy()

        if model_state_dict is not None:
            learner.model.load_state_dict(model_state_dict)
        test_loss = learner.valid_epoch(learner.test_iter)
        self.logger.info(f"Trial {trial.number} | Test-Loss = {test_loss}")
        return test_loss

    def run(self, objective, trials: int = 100, start_up: int = 0,
            warm_up: int = 0, epochs: int = 1,
            num_train_samples: int | float = 0, num_test_samples: int | float = 0,
            load_vocabularies: bool = False, simplified_tokens: bool = False,
            use_extra_test_samples: bool = False, process_count: int = 1,
            disable_backend: bool = False, debug: bool = False) -> optuna.Study:

        self.param_history.clear()
        token_type = "simplified" if simplified_tokens else "enumerated"
        study_type = "model" if objective == self.__objective_model__ else "learning"
        objective_str = "Model Parameter Search" if objective == self.__objective_model__ else "Learning Rate Search"
        self.name = f"{objective_str} | {self.source} -> {self.target} | Type: {'simplified' if simplified_tokens else 'enumerated'}"
        log_filename = f"parameter_search_{self.source}_{self.target}_{token_type}.log"
        self.logger = Logger(LoggingLevel.INFO, self.params_dir, log_filename) if debug \
            else FileLogger(log_filename[:-4], LoggingLevel.INFO, self.params_dir, log_filename)

        self.trials = trials if trials > 0 else 1
        self.start_up = start_up if start_up > 0 else 0
        self.warm_up = warm_up if warm_up > 0 else 0
        self.process_count = process_count if process_count > 1 else 1
        self.num_train_samples = num_train_samples if num_train_samples > 0 else 0
        self.num_test_samples = num_test_samples if num_test_samples > 0 else 0
        self.epochs = epochs if epochs > 0 else 1
        self.debug_mode = debug
        self.params["disable_backend"] = disable_backend
        self.params["simplified"] = simplified_tokens
        self.load_vocabularies = load_vocabularies
        self.simplified_tokens = simplified_tokens
        self.use_extra_test_samples = use_extra_test_samples

        pruner = optuna.pruners.MedianPruner(n_startup_trials=self.start_up, n_warmup_steps=self.warm_up)
        study = optuna.create_study(direction="minimize", pruner=pruner, study_name=self.name, sampler=optuna.samplers.GridSampler(self.config))
        if self.debug_mode:
            self.logger.info(f"Searching for optimal {study_type} hyperparameters | {self.source} -> {self.target} | Type: {token_type}")
            self.logger.info(f"Total Trials: {self.trials} | Start-up Trials: {self.start_up} | Warm-up Trials: {self.warm_up}")
        if self.process_count > 1:
            with parallel_backend('multiprocessing'):
                study.optimize(objective, n_trials=self.trials, n_jobs=self.process_count)
        else:
            study.optimize(objective, n_trials=self.trials)
        for k, v in study.best_params.items():
            self.params[k] = v
        pickle_dump(self.params, self.params_dir.joinpath(f"hyperparameters_{study_type}_{self.source}_{self.target}_{token_type}.pickle").resolve())
        pickle_dump(study, self.params_dir.joinpath(f"study_{study_type}_{self.source}_{self.target}_{token_type}.pkl").resolve())
        study.trials_dataframe().to_csv(self.params_dir.joinpath(f"trials_{study_type}_{self.source}_{self.target}_{token_type}.csv").resolve())

        message1 = f"Best {study_type.capitalize()} Parameters: {study.best_params}"
        message2 = f"Best {study_type.capitalize()} Loss: {study.best_value}"
        linebreak = f"Search completed\n{'—' * 250}"

        if self.debug_mode:
            self.logger.info(message1)
            self.logger.info(message2)
            self.logger.info(linebreak)
        else:
            print(message1)
            print(message2)
            print(linebreak)
        return study

    def run_model_params_search(self, trials: int = 100, start_up: int = 0,
                                warm_up: int = 0, epochs: int = 1,
                                num_train_samples: int | float = 0, num_test_samples: int | float = 0,
                                load_vocabularies: bool = False, simplified_tokens: bool = False,
                                use_extra_test_samples: bool = False, process_count: int = 1,
                                disable_backend: bool = False, debug: bool = False) -> optuna.Study:
        args = list(locals().values())[1:]
        return self.run(self.__objective_model__, *args)

    def run_learning_rate_search(self, trials: int = 100, start_up: int = 0,
                                 warm_up: int = 0, epochs: int = 1,
                                 num_train_samples: int | float = 0, num_test_samples: int | float = 0,
                                 load_vocabularies: bool = False, simplified_tokens: bool = False,
                                 use_extra_test_samples: bool = False, process_count: int = 1,
                                 disable_backend: bool = False, debug: bool = False) -> optuna.Study:
        args = list(locals().values())[1:]
        token_type = "simplified" if simplified_tokens else "enumerated"
        f_path = self.params_dir.joinpath(f"hyperparameters_model_{self.source}_{self.target}_{token_type}.pickle")
        loaded = pickle_load(f_path.resolve())
        self.params = Learner.HYPERPARAMETERS.copy() if loaded is None else loaded
        print(f"Loaded Parameters: {self.params}")
        return self.run(self.__objective_learning__, *args)
