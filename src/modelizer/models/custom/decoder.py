import torch

from pathlib import Path
from copy import deepcopy
from typing import Any, Optional

from pandas import DataFrame

from modelizer import configs
from modelizer.models import dataset
from modelizer.utils import Logger, load_module
from modelizer.configs import Template, AUTOREGRESSIVE_TEMPLATE_BLANK
from x_transformers import AutoregressiveWrapper, TransformerWrapper, Decoder
from modelizer.models.custom.custom import CustomConfig, CustomModel, BaseTokenizer


########################################################################################################################
#                                      Configuration for Decoder-only Models                                           #
########################################################################################################################
class DecoderConfig(CustomConfig):
    """Configuration class for the Transformer Decoder model"""

    def check_constraints(self):
        if self._feedforward_size == 0:
            self._feedforward_size = self._embedding_size * 2
        if self._hidden_size == 0:
            self._hidden_size = self._embedding_size

    def __init__(self,
                 output_dir: str | Path,
                 source: str,
                 target: str,
                 backward: bool,
                 embedding_size: int,
                 num_heads: int,
                 vocab_size: int,
                 num_layers: int,
                 max_sequence_length: int,
                 *,
                 feedforward_size: int = 0,
                 hidden_size: int = 0,
                 dropout: float = configs.DROPOUT,
                 layer_dropout: float = configs.LAYER_DROPOUT,
                 attn_dropout: float = configs.ATTN_DROPOUT,
                 ff_dropout: float = configs.FF_DROPOUT,
                 attn_flash: bool = True,
                 optimizer: str = configs.OPTIMIZER,
                 scheduler: str = configs.SCHEDULER,
                 learning_rate: float = configs.LEARNING_RATE,
                 weight_decay: float = configs.WEIGHT_DECAY,
                 b1: float = configs.B1,
                 b2: float = configs.B2,
                 eps: float = configs.EPS,
                 clip_grad: float = configs.CLIP_GRAD,
                 validation_fraction: float = configs.VALIDATION_FRACTION,
                 validation_overlap: bool = False,
                 checkpoint_interval: int = configs.CHECKPOINT_INTERVAL,
                 use_flash: bool = True,
                 compile_model: bool = False,
                 force_cpu: bool = False,
                 reduce_memory_usage: bool = False,
                 free_cached_memory: bool = False,
                 activation: str = "gelu",
                 positional_encoding_type: str = "sinusoidal",
                 instructions: Optional[Template] = AUTOREGRESSIVE_TEMPLATE_BLANK,
                 total_save_limit: int = configs.TOTAL_SAVE_LIMIT,
                 wandb_token: Optional[str] = None,
                 seed: int = configs.SEED,
                 reduce_spaces: bool = False,
                 metadata: Optional[dict[str, Any]] = None, **_):
        """
        Constructs the decoder-only model configuration.
        :param output_dir: Directory to save the model and tokenizer.
        :param source: Source type.
        :param target: Target type.
        :param backward: If True, the model will be trained in the backward direction.
        :param embedding_size: Embedding dimension for the decoder.
        :param feedforward_size: Feed-forward dimension for the decoder. Only relevant for legacy models. Default is 0, which maps to 2x embedding_size.
        :param hidden_size: Hidden layer dimension size. Default is 0, which means it is equal to embedding_size. Affects legacy models.
        :param num_heads: Number of attention heads in the decoder.
        :param vocab_size: Vocabulary size for the decoder.
        :param num_layers: Number of decoder layers.
        :param max_sequence_length: Maximum sequence length for the decoder.

        :param dropout: Embedding dropout probability. Default is configs.DROPOUT.
        :param layer_dropout: Layer dropout probability. Default is configs.LAYER_DROPOUT.
        :param attn_dropout: Attention dropout probability. Default is configs.ATTN_DROPOUT.
        :param ff_dropout: Feed-forward dropout probability. Default is configs.FF_DROPOUT.
        :param attn_flash: If True, the model will be trained in the flash direction.
        :param optimizer: Optimizer to use for training. Default is configs.OPTIMIZER.
                          Possible values are 'sgd', 'adam', 'adamw', 'adagrad', 'paged_adamw', 'rmsprop',
                          'sgd_8bit', 'adam_8bit', 'adamw_8bit', 'adagrad_8bit', 'paged_adamw_8bit', 'rmsprop_8bit'.
                          8bit versions only function on systems with CUDA-enabled PyTorch installation.
        :param scheduler: Learning rate scheduler to use for training. Default is configs.SCHEDULER.
                          Possible values: 'linear', 'cyclic', 'cosine', 'step', 'polynomial', None
        :param learning_rate: Learning rate for the optimizer. Default is configs.LEARNING_RATE.
        :param weight_decay: Weight decay for the optimizer. Default is configs.WEIGHT_DECAY.
        :param b1: Beta1 parameter for the optimizer. Default is configs.B1.
        :param b2: Beta2 parameter for the optimizer. Default is configs.B2.
        :param eps: Epsilon parameter for the optimizer. Default is configs.EPS.
        :param clip_grad: Gradient clipping value. Default is configs.CLIP_GRAD.
        :param validation_fraction: Training dataset fraction used to validate the model during training. Default is configs.VALIDATION_FRACTION.
        :param validation_overlap: If True, the validation set will overlap with the training set. Default is False.
        :param checkpoint_interval: Interval for saving checkpoints. Default is configs.CHECKPOINT_INTERVAL.
        :param use_flash: If True, use the FLASH attention mechanism in Transformer. Default is True. Affects legacy models.
        :param compile_model: If True, compile the model before training. Default is False.
        :param force_cpu: If True, force the model to run on CPU. Default is False.
        :param reduce_memory_usage: If True, the model will reduce parameter's precision to save memory. Default is False.
        :param free_cached_memory: If True, free cached memory after each epoch.
        :param activation: Activation function for Transformer layers. Default is 'gelu'. Affects legacy models. Possible values 'relu' or 'gelu'.
        :param positional_encoding_type: Type of positional encoding to use. Default is 'sinusoidal'. Affects legacy models. Can be 'sinusoidal', 'learnable', 'rope'.
        :param instructions: (Optional) Prompt template for the model. Default is templates.AUTOREGRESSIVE_TEMPLATE_BLANK.
        :param total_save_limit: The maximum number of model checkpoints to keep. Default is configs.TOTAL_SAVE_LIMIT.
        :param wandb_token: (Optional) Weights and Biases API token. Default is None.
        :param seed: Random seed for reproducibility, Default is configs.SEED.
        :param reduce_spaces: If True, the model will reduce spaces in the input data.
        :param metadata: Optional metadata dictionary to store additional information about the model.
        """
        kwargs = locals()
        kwargs.pop('self')
        kwargs.pop("__class__", None)
        super().__init__(**kwargs)
        # Model Configs
        self._embedding_size = embedding_size
        self._feedforward_size = feedforward_size if feedforward_size > 0 else embedding_size * 2
        self._hidden_size = hidden_size if hidden_size > 0 else embedding_size
        self._vocab_size = vocab_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._max_sequence_length = max_sequence_length
        self._use_flash = use_flash
        self._activation = activation
        self._positional_encoding_type = positional_encoding_type
        self._post_emb_norm = False
        # Dropout Configs
        self._dropout = dropout
        self._layer_dropout = layer_dropout
        self._attn_dropout = attn_dropout
        self._ff_dropout = ff_dropout
        # Feed Forward Configs
        self._ff_glu = True
        self._ff_swish = True
        self._ff_no_bias = True
        # Attention Configs
        self._attn_flash = attn_flash
        self._attn_one_kv_head = False
        self._gate_residual = False
        # Layer Normalization Configs
        self._use_simple_rmsnorm = True
        self._sandwich_norm = False
        # Positional Encoding Configs
        self._rel_pos_bias = False
        self._rotary_pos_emb = True
        self._rotary_xpos = False
        self._use_abs_pos_emb = True
        self._use_abs_pos_norm = False
        self._report_epoch_progress = False

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def feedforward_size(self) -> int:
        return self._feedforward_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def use_flash(self) -> bool:
        return self._use_flash

    @property
    def activation(self) -> str:
        return self._activation

    @property
    def positional_encoding_type(self) -> str:
        return self._positional_encoding_type

    @property
    def post_emb_norm(self) -> bool:
        return self._post_emb_norm

    @post_emb_norm.setter
    def post_emb_norm(self, value: bool):
        assert isinstance(value, bool), "post_emb_norm must be a boolean value"
        self._post_emb_norm = value

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def layer_dropout(self) -> float:
        return self._layer_dropout

    @property
    def attn_dropout(self) -> float:
        return self._attn_dropout

    @property
    def ff_dropout(self) -> float:
        return self._ff_dropout

    @property
    def ff_glu(self) -> bool:
        return self._ff_glu

    @ff_glu.setter
    def ff_glu(self, value: bool):
        assert isinstance(value, bool), "ff_glu must be a boolean value"
        self._ff_glu = value

    @property
    def ff_swish(self) -> bool:
        return self._ff_swish

    @ff_swish.setter
    def ff_swish(self, value: bool):
        assert isinstance(value, bool), "ff_swish must be a boolean value"
        self._ff_swish = value

    @property
    def ff_no_bias(self) -> bool:
        return self._ff_no_bias

    @ff_no_bias.setter
    def ff_no_bias(self, value: bool):
        assert isinstance(value, bool), "ff_no_bias must be a boolean value"
        self._ff_no_bias = value

    @property
    def attn_flash(self) -> bool:
        return self._attn_flash

    @property
    def attn_one_kv_head(self) -> bool:
        return self._attn_one_kv_head

    @property
    def gate_residual(self) -> bool:
        return self._gate_residual

    @attn_flash.setter
    def attn_flash(self, value: bool):
        assert isinstance(value, bool), "attn_flash must be a boolean value"
        self._attn_flash = value

    @property
    def use_simple_rmsnorm(self) -> bool:
        return self._use_simple_rmsnorm

    @use_simple_rmsnorm.setter
    def use_simple_rmsnorm(self, value: bool):
        assert isinstance(value, bool), "use_simple_rmsnorm must be a boolean value"
        self._use_simple_rmsnorm = value

    @property
    def sandwich_norm(self) -> bool:
        return self._sandwich_norm

    @property
    def rel_pos_bias(self) -> bool:
        return self._rel_pos_bias

    @property
    def rotary_pos_emb(self) -> bool:
        return self._rotary_pos_emb

    @property
    def rotary_xpos(self) -> bool:
        return self._rotary_xpos

    @property
    def use_abs_pos_emb(self) -> bool:
        return self._use_abs_pos_emb

    @property
    def use_abs_pos_norm(self) -> bool:
        return self._use_abs_pos_norm

    @property
    def report_epoch_progress(self) -> bool:
        return self._report_epoch_progress

    @report_epoch_progress.setter
    def report_epoch_progress(self, value: bool):
        assert isinstance(value, bool), "report_epoch_progress must be a boolean"
        self._report_epoch_progress = value

    def get_configurable_parameters(self, force_cpu: bool = False) -> dict[str, list[Any]]:
        config = deepcopy(configs.XTR_DECODER_PARAMETERS)
        if torch.cuda.is_available() and not force_cpu and not self.cross_platform_compatibility:
            config["optimizer"].extend([f"{opt}_8bit" for opt in config["optimizer"]])
        return config


########################################################################################################################
#                                               Decoder-only Model                                                     #
########################################################################################################################
class DecoderModel(CustomModel):
    """GPT-like Decoder-only custom transformer model."""

    def __init__(self, config: DecoderConfig, tokenizer: BaseTokenizer, logger: Optional[Logger] = None, force_cpu: bool = False, **_):
        assert isinstance(config, DecoderConfig), "config must be an instance of Config"
        config.check_constraints()
        flash_runtime_supported = torch.cuda.is_available() and not force_cpu and not getattr(config, "force_cpu", False)
        model = TransformerWrapper(
            num_tokens=config.vocab_size,
            max_seq_len=config.max_sequence_length,
            use_abs_pos_emb=config.use_abs_pos_emb,
            emb_dropout=config.dropout,
            attn_layers=Decoder(
                dim=config.embedding_size,
                depth=config.num_layers,
                heads=config.num_heads,
                ff_glu=config.ff_glu,
                ff_swish=config.ff_swish,
                ff_no_bias=config.ff_no_bias,
                attn_flash=config.attn_flash and flash_runtime_supported,
                attn_one_kv_head=config.attn_one_kv_head,
                gate_residual=config.gate_residual,
                rel_pos_bias=config.rel_pos_bias,
                rotary_xpos=config.rotary_xpos,
                rotary_pos_emb=config.rotary_pos_emb,
                use_simple_rmsnorm=config.use_simple_rmsnorm,
                sandwich_norm=config.sandwich_norm,
                layer_dropout=config.layer_dropout,
                attn_dropout=config.attn_dropout,
                ff_dropout=config.ff_dropout,
            )
        )
        model = AutoregressiveWrapper(
            model,
            ignore_index=tokenizer.pad_token_id,
            pad_value=tokenizer.pad_token_id,
        )
        super().__init__(model, config, tokenizer, logger=logger, force_cpu=force_cpu)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self._tokenizer.pad_token_id)
        self.__instructions__ = Template(self.config.instructions.safe_substitute({"cls": self._tokenizer.cls_token}))

    def __forge_dataset__(self, dataframe: dataset.DataFrame) -> dataset.TorchDataset:
        """
        This method is responsible for creating a TorchDataset from the given DataFrame. Do not call this method directly.
        :param dataframe: pandas DataFrame object
        :return: TorchAutoRegressiveDataset object
        """
        return dataset.TorchAutoRegressiveDataset(dataframe, self.config.source, self.config.target, self._tokenizer, self.__instructions__)

    def __forward__(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This method is responsible for passing the input through the model and returning the loss.
        Do not call this method directly. It is called by train, test methods.
        :param sample: dict[str, torch.Tensor] containing input_ids, attention_mask (input padding mask)
        """
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"].bool() if "attention_mask" in sample else None
        target_mask = attention_mask[:, 1:] if attention_mask is not None else input_ids[:, 1:].ne(
            self._tokenizer.pad_token_id)
        if not torch.any(target_mask):
            raise ValueError("Decoder batches must contain at least one non-padding next-token target.")
        model_kwargs = {}
        if attention_mask is not None:
            model_kwargs["mask"] = attention_mask
        return self._model(input_ids, **model_kwargs)

    def check_tokenizers(self, *, dataframe: DataFrame, **kwargs):
        input_tokenizer_path = kwargs.get("input_tokenizer_path", self.config.tokenizer_path)
        self.__check_input_tokenizer__(input_tokenizer_path, dataframe)

    @torch.inference_mode()
    def generate(self, input_data: Any, max_length: int = 256, **_) -> Any:
        """
        Generate output from the input data.
        :param input_data: Vector of input data or a single string
        :param max_length: Maximum length of the output
        :return: Generated output as a string or a vector
        """
        input_data = self.__instructions__.substitute({"input": input_data, "response": ""})
        tokenized_input = self._tokenizer(input_data)["input_ids"][:-1]
        output = self._model.generate(prompts=tokenized_input, seq_len=max_length, cache_kv=True)
        output = self.filter_cls_token(output)
        return self._tokenizer.reconstruct(output)

    @staticmethod
    def from_pretrained(filepath: str | Path, logger: Optional[Logger] = None) -> "DecoderModel":
        filepath = CustomModel.check_model_filepath(filepath, logger=logger)
        config = CustomConfig.from_pretrained(filepath, logger)
        tokenizer = load_module(config.output_dir.joinpath(config.tokenizer_path), logger)
        return DecoderModel(config, tokenizer, logger)
