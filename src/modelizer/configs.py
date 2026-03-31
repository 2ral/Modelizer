from string import Template

AUTOREGRESSIVE_TEMPLATE = Template("$input <|cls|> $response")
AUTOREGRESSIVE_TEMPLATE_BLANK = Template("$input $cls $response")

SEED = 527697                            # Default seed for random module
TIMEOUT_SECONDS = 10                     # Default timeout for test execution
SPACE_TOKEN = "<|spc|>"                  # Token representing space character
UNKNOWN_FEATURE = "<|UNKNOWN_FEATURE|>"  # Token representing unknown feature in feature models

MODELIZER_GENERATOR_CACHE_SIZE = 128     # Cache size for the generate method in Modelizer models

TOTAL_SAVE_LIMIT = 3                     # Maximum number of checkpoints to keep
VALIDATION_FRACTION = 0.2                # Fraction of data used for validation
CHECKPOINT_INTERVAL = 1000000            # Interval for saving checkpoints

OPTIMIZER = "adamw"                      # Default optimizer for the model training
SCHEDULER = None                         # Default learning rate scheduler for the optimizer
CLIP_GRAD = 1.0                          # Gradient clipping value
LEARNING_RATE = 1e-5                     # Default learning rate for the optimizer
WEIGHT_DECAY = 1e-3                      # Default weight decay for the optimizer
B1 = 0.9                                 # Beta1 parameter for the optimizer
B2 = 0.999                               # Beta2 parameter for the optimizer
EPS = 1e-8                               # Epsilon parameter for the optimizer

DROPOUT = 0.1                            # Default dropout rate for the model
LAYER_DROPOUT = 0.                       # Default layer dropout rate for the model
ATTN_DROPOUT = 0.                        # Default attention dropout rate for the model
FF_DROPOUT = 0.                          # Default feedforward dropout rate for the model


# Base configs for reuse
BASE_OPTIMIZERS = ["adamw", "sgd", "rmsprop"]
BASE_SCHEDULERS = ["linear", "cyclic", "cosine", "step", "polynomial", None]
BASE_WEIGHT_DECAY = [0.01, 0.05, 0.1, 0.2]
BASE_LEARNING_RATE = [5e-5, 1e-4, 2e-4, 3e-4, 5e-4]
BASE_EMBEDDING_SIZES = [256, 512, 1024, 2048, 4096]
BASE_HIDDEN_SIZES = [0, 256, 512, 1024, 2048, 4096]
BASE_NUM_LAYERS = [1, 2, 3, 4, 5, 6, 8]
BASE_NUM_HEADS = [4, 8, 16, 32, 64]
BASE_FEEDFORWARD_SIZES = [0, 256, 512, 1024, 2048, 4096]
BASE_DROPOUT_RATES = [0.0, 0.1, 0.2]
BASE_CLIP_GRAD_RATES = [None, 1.0]
CNN_NUM_LAYERS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
CNN_KERNEL_SIZES = [3, 5, 7, 9, 11]


# LEGACY_MODEL_PARAMETERS
LEGACY_MODEL_PARAMETERS = {
    "embedding_size": BASE_EMBEDDING_SIZES,
    "num_heads": BASE_NUM_HEADS,
    "feedforward_size": BASE_FEEDFORWARD_SIZES,
    "enc_layers": BASE_NUM_LAYERS,
    "dec_layers": BASE_NUM_LAYERS,
    "dropout": BASE_DROPOUT_RATES,
    "clip_grad": BASE_CLIP_GRAD_RATES,
    "optimizer": BASE_OPTIMIZERS,
    "scheduler": [None, "linear", "lambda", "multiplicative", "cosine", "step", "exponential"],
    "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3],
    "weight_decay": BASE_WEIGHT_DECAY,
}

# XTR_PARAMETERS
XTR_PARAMETERS = {
    "embedding_size": BASE_EMBEDDING_SIZES,
    "num_heads": BASE_NUM_HEADS,
    "enc_layers": BASE_NUM_LAYERS,
    "dec_layers": BASE_NUM_LAYERS,
    "enc_dropout": BASE_DROPOUT_RATES,
    "enc_layer_dropout": BASE_DROPOUT_RATES,
    "enc_attn_dropout": BASE_DROPOUT_RATES,
    "enc_ff_dropout": BASE_DROPOUT_RATES,
    "dec_dropout": BASE_DROPOUT_RATES,
    "dec_layer_dropout": BASE_DROPOUT_RATES,
    "dec_attn_dropout": BASE_DROPOUT_RATES,
    "dec_ff_dropout": BASE_DROPOUT_RATES,
    "clip_grad": BASE_CLIP_GRAD_RATES,
    "optimizer": BASE_OPTIMIZERS,
    "scheduler": BASE_SCHEDULERS,
    "learning_rate": BASE_LEARNING_RATE,
    "weight_decay": BASE_WEIGHT_DECAY,
}
