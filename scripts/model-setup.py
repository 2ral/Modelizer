if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from sys import path as sys_path
    sys_path[0] = Path(__file__).resolve().parent.parent.as_posix()
    from modelizer.optimizer import ParameterSearchOptimizer

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train', type=str, default="datasets/train/data_nop_10k", help='Path to Directory containing Train Data')
    arg_parser.add_argument('--test', type=str, default="datasets/test/test_10k", help='Optional Path to Directory containing Test Data')
    arg_parser.add_argument('--num-train-samples', type=int, default=0, help='Specify the Number of Train Samples to Use')
    arg_parser.add_argument('--num-test-samples', type=int, default=0, help='Specify the Number of Test Samples to Use')
    arg_parser.add_argument('-s', '--source', type=str, default="html", help='Source Data Type for Hyperparameter Search')
    arg_parser.add_argument('-t', '--target', type=str, default="markdown", help='Target Data Type for Hyperparameter Search')
    arg_parser.add_argument('--cpu', action='store_true', help='Run using CPU')
    arg_parser.add_argument('--extra', action='store_true', help='Add extra test cases to the test set')
    arg_parser.add_argument('--debug', action='store_true', help='Enable Debug Print Statements')
    arg_parser.add_argument('--simplified', action='store_true', help='Switch to Simplified Token Representation')
    arg_parser.add_argument('--process-count', type=int, default=1, help='Number of Parallel Processes to Use')
    arg_parser.add_argument('--param-trials', type=int, default=100, help='Total Number of Param-Search Trials')
    arg_parser.add_argument('--lr-trials', type=int, default=100, help='Total Number of LR-Search Trials')
    parsed_args = arg_parser.parse_args()

    # 0. Setup
    optimizer = ParameterSearchOptimizer(parsed_args.source.lower(), parsed_args.target.lower(), parsed_args.train, parsed_args.test)
    optimizer.config = {
        "feedforward_size": [1024, 2048, 4096],
        "embedding_size": [256, 512, 1024],
        "head_count": [16, 32, 64],
        "num_encoder_layers": [1, 2, 3, 4],
        "num_decoder_layers": [1, 2, 3, 4],
        "clip_gradients": [None, 1.0],
        "dropout": [0.0, 0.1],
        "learning_policy": [None, "lambda", "multiplicative", "cosine", "step", "exponential"],
        "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01],
        "weight_decay": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    }

    # 1. Model Hyperparameter Search
    if parsed_args.param_trials > 0:
        optimizer.run_model_params_search(parsed_args.param_trials, parsed_args.param_trials,
                                          0, 4,
                                          parsed_args.num_train_samples, parsed_args.num_test_samples,
                                          True, parsed_args.simplified,
                                          parsed_args.extra, parsed_args.process_count,
                                          parsed_args.cpu, parsed_args.debug)

    # 2. Learning Rate Search
    if parsed_args.lr_trials > 0:
        optimizer.run_learning_rate_search(parsed_args.lr_trials, parsed_args.lr_trials,
                                           0, 5,
                                           parsed_args.num_train_samples, parsed_args.num_test_samples,
                                           True, parsed_args.simplified,
                                           parsed_args.extra, parsed_args.process_count,
                                           parsed_args.cpu, parsed_args.debug)
