if __name__ == "__main__":
    from pathlib import Path
    from datetime import datetime
    from sys import path as sys_path
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()
    from modelizer.trainer import get_training_args
    from modelizer.learner import Learner
    from modelizer.utils import Logger, LoggingLevel, infer_subject, pickle_load
    from modelizer.dataset import TrainDataset, TorchDataset, load_vocabs, save_vocabs

    start_time = datetime.now()
    args = get_training_args()
    source = args.source.lower()
    target = args.target.lower()
    train_data_dir = Path(args.train).resolve()
    assert train_data_dir.exists(), f"Data Directory {train_data_dir.as_posix()} does not exist"
    params_dir = train_data_dir.parent.joinpath("hyperparameters")
    root_dir = train_data_dir.joinpath("models").joinpath(f"modelizer_simplified_{source}_{target}" if args.simplified else f"modelizer_{source}_{target}")
    root_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(LoggingLevel.DEBUG if args.enable_debug_logging else LoggingLevel.INFO, root_dir.as_posix())
    logger.info(f"Task: Model Training | Date and Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    params = Learner.HYPERPARAMETERS.copy()
    if args.params:
        param_file_candidates = [
            params_dir.joinpath(f"hyperparameters_learning_{source}_{target}_{'simplified' if args.simplified else 'enumerated'}.pickle"),
            params_dir.joinpath(f"hyperparameters_learning_{target}_{source}_{'simplified' if args.simplified else 'enumerated'}.pickle"),
            params_dir.joinpath(f"hyperparameters_model_{source}_{target}_{'simplified' if args.simplified else 'enumerated'}.pickle"),
            params_dir.joinpath(f"hyperparameters_model_{target}_{source}_{'simplified' if args.simplified else 'enumerated'}.pickle")
        ]

        for filepath in param_file_candidates:
            loaded = pickle_load(filepath)
            if loaded is not None:
                logger.info(f"Loaded Hyperparameter Configuration: {filepath.as_posix()}")
                params.update(loaded)
                break

    params["epoch"] = 0
    params["source"] = source
    params["target"] = target
    params["debug"] = args.enable_debug_logging
    params["disable_backend"] = args.cpu
    params["free_cached_memory"] = args.clear
    params["pos_encoding_size"] = args.pos_encoding_size

    subject = infer_subject(source, target)
    dataset_filename_filter = "simplified.*" if args.simplified else "dataset.*"
    vocabs_dir = train_data_dir.parent.joinpath("vocabs").joinpath(f"{subject}_simplified" if args.simplified else subject)
    vocabularies = load_vocabs(vocabs_dir) if args.vocab else None
    if vocabularies is not None:
        logger.info(f"Vocabularies loaded from {vocabs_dir.as_posix()}")

    logger.info("Loading Dataset...")
    train_data_dir = train_data_dir.joinpath(subject)
    assert train_data_dir.exists(), f"Train Data Directory {train_data_dir.as_posix()} does not exist"
    assert not train_data_dir.is_file(), f"Train Data Directory path {train_data_dir.as_posix()} points to a file"
    logger.info(f"Train Data Directory: {train_data_dir.as_posix()}")
    train_dataset = TrainDataset(list(train_data_dir.glob(dataset_filename_filter))[0], source, target, vocabularies, args.num_train_samples)
    logger.info(f"Dataset loaded in {datetime.now() - start_time} | Records: {len(train_dataset)}")

    if len(args.test):
        test_loading_start = datetime.now()
        test_data_dir = Path(args.test).resolve()
        if subject not in args.test:
            test_data_dir = test_data_dir.joinpath(subject)
        assert test_data_dir.exists(), f"Test Data Directory {test_data_dir.as_posix()} does not exist"
        assert not test_data_dir.is_file(), f"Test Data Directory path {test_data_dir.as_posix()} points to a file"
        logger.info(f"Test Data Directory: {test_data_dir.as_posix()}")
        test_dataset = TorchDataset(list(test_data_dir.glob(dataset_filename_filter))[0], source, target, train_dataset.vocabs, args.num_test_samples)
        test_data_loader = test_dataset.get_dataloader(batch_size=args.batch_size, shuffle=args.shuffle_dataset)
        logger.info(f"Dataset loaded in {datetime.now() - test_loading_start} | Records: {len(test_data_loader)}")
        train_d, eval_d, _ = train_dataset.get_dataloaders(args.train_fraction, 0.0, batch_size=args.batch_size, shuffle=args.shuffle_dataset)
        dataloaders = (train_d, eval_d, test_data_loader)
    else:
        dataloaders = train_dataset.get_dataloaders(args.train_fraction, args.test_fraction, batch_size=args.batch_size, shuffle=args.shuffle_dataset)

    logger.info(f"Source: {source} | Target: {target} | Initializing the Model...")
    learner = Learner(dataloaders=dataloaders, vocabularies=train_dataset.get_vocabularies(), params=params, logger=logger)
    save_vocabs(train_dataset.vocabs, root_dir)
    logger.info(f"Transformer model initialized in {datetime.now() - start_time}")
    logger.info(f"Training the Model for {args.epochs} epochs")
    result = learner.train(args.epochs, root_dir.joinpath("{}.{}").as_posix())
    logger.info(f"Model trained in {datetime.now() - start_time}" if result is not None else "Model training failed")
