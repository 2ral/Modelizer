import torch

from x_transformers import XTransformer

from pathlib import Path
from copy import deepcopy
from typing import Any, Optional

from modelizer import configs
from modelizer.models import dataset
from modelizer.utils import Logger, load_module
from modelizer.tokenizers.abstract import BaseTokenizer
from modelizer.models.custom.custom import CustomModel, CustomConfig


########################################################################################################################
#                                Configuration for complex Encoder-Decoder Models                                      #
########################################################################################################################
class EncoderDecoderConfig(CustomConfig):
    """Configuration class for complex Transformer Encoder-Decoder models"""

    def __init__(self,
                 output_dir: str | Path,
                 source: str,
                 target: str,
                 backward: bool,
                 embedding_size: int,
                 num_heads: int,
                 source_vocab_size: int,
                 enc_layers: int,
                 source_max_len: int,
                 target_vocab_size: int,
                 dec_layers: int,
                 target_max_len: int,
                 *,
                 feedforward_size: int = 0,
                 enc_dropout: float = 0.,
                 enc_layer_dropout: float = 0.,
                 enc_attn_dropout: float = 0.,
                 enc_ff_dropout: float = 0.,
                 dec_dropout: float = 0.,
                 dec_layer_dropout: float = 0.,
                 dec_attn_dropout: float = 0.,
                 dec_ff_dropout: float = 0.,
                 enc_attn_flash: bool = True,
                 dec_attn_flash: bool = True,
                 optimizer: str = configs.OPTIMIZER,
                 scheduler: Optional[str] = configs.SCHEDULER,
                 learning_rate: float = configs.LEARNING_RATE,
                 weight_decay: float = configs.WEIGHT_DECAY,
                 b1: float = configs.B1,
                 b2: float = configs.B2,
                 eps: float = configs.EPS,
                 clip_grad: float = configs.CLIP_GRAD,
                 validation_fraction: float = configs.VALIDATION_FRACTION,
                 checkpoint_interval: int = configs.CHECKPOINT_INTERVAL,
                 use_flash: bool = True,
                 compile_model: bool = False,
                 force_cpu: bool = False,
                 reduce_memory_usage: bool = False,
                 free_cached_memory: bool = False,
                 hidden_size: int = 0,
                 activation: str = "gelu",
                 positional_encoding_type: str = "sinusoidal",
                 total_save_limit: int = configs.TOTAL_SAVE_LIMIT,
                 wandb_token: Optional[str] = None,
                 seed: int = configs.SEED,
                 reduce_spaces: bool = False,
                 metadata: Optional[dict[str, Any]] = None, **_):
        """
        Constructs the sequence-to-sequence model configuration.
        :param output_dir: Directory to save the model and tokenizer.
        :param source: Source type.
        :param target: Target type.
        :param backward: If True, the model will be trained in the backward direction.
        :param embedding_size: Embedding dimension.
        :param feedforward_size: Feed-forward dimension. Only relevant for legacy models. Default is 0, which maps to x2 embedding_size.
        :param hidden_size: Hidden layer dimension size. Default is 0, which means it is equal to embedding_size. Affects legacy models.
        :param num_heads: Number of attention heads.
        :param source_vocab_size: Vocabulary size for the encoder.
        :param enc_layers: Number of encoder layers.
        :param source_max_len: Maximum sequence length for the encoder.
        :param target_vocab_size: Vocabulary size for the decoder.
        :param dec_layers: Number of decoder layers.
        :param target_max_len: Maximum sequence length for the decoder.
        :param enc_dropout: Embedding dropout probability for the encoder. Default is 0.
        :param enc_layer_dropout: Layer dropout probability for the encoder. Default is 0.
        :param enc_attn_dropout: Attention dropout probability for the encoder. Default is 0.
        :param enc_ff_dropout: Feed-forward dropout probability for the encoder. Default is 0.
        :param dec_dropout: Embedding dropout probability for the decoder. Default is 0.
        :param dec_layer_dropout: Layer dropout probability for the decoder. Default is 0.
        :param dec_attn_dropout: Attention dropout probability for the decoder. Default is 0.
        :param dec_ff_dropout: Feed-forward dropout probability for the decoder. Default is 0.
        :param enc_attn_flash: If True, the model will be trained in the flash direction.
        :param dec_attn_flash: If True, the model will be trained in the flash direction.
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
        :param checkpoint_interval: Interval for saving checkpoints. Default is configs.CHECKPOINT_INTERVAL.
        :param use_vanilla: If True, use the vanilla implementation of Transformer model. Default is False.
        :param use_flash: If True, use the FLASH attention mechanism in Transformer. Default is True. Affects legacy models.
        :param compile_model: If True, compile the model before training. Default is False.
        :param force_cpu: If True, force the model to run on CPU. Default is False.
        :param reduce_memory_usage: If True, the model will reduce parameter's precision to save memory. Default is False.
        :param free_cached_memory: If True, free cached memory after each epoch.
        :param hidden_size: Hidden layer dimension size. Default is 0, which means it is equal to embedding_size. Affects legacy models.
        :param activation: Activation function for Transformer layers. Default is 'gelu'. Affects legacy models. Possible values 'relu' or 'gelu'.
        :param positional_encoding_type: Type of positional encoding to use. Default is 'sinusoidal'. Affects legacy models. Can be 'sinusoidal', 'learnable', 'rope'.
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
        self._embedding_size = embedding_size
        self._feedforward_size = feedforward_size if feedforward_size > 0 else embedding_size * 2
        self._hidden_size = hidden_size if hidden_size > 0 else embedding_size
        self._num_heads = num_heads
        # Encoder Configs
        self._enc_vocab_size = source_vocab_size
        self._enc_layers = enc_layers
        self._source_max_len = source_max_len
        # Decoder Configs
        self._dec_vocab_size = target_vocab_size
        self._dec_layers = dec_layers
        self._target_max_len = target_max_len
        # Encoder Dropout Configs
        self._enc_dropout = enc_dropout
        self._enc_layer_dropout = enc_layer_dropout
        self._enc_attn_dropout = enc_attn_dropout
        self._enc_ff_dropout = enc_ff_dropout
        # Decoder Dropout Configs
        self._dec_dropout = dec_dropout
        self._dec_layer_dropout = dec_layer_dropout
        self._dec_attn_dropout = dec_attn_dropout
        self._dec_ff_dropout = dec_ff_dropout
        # Encoder Feed Forward Configs
        self._enc_ff_glu = True
        self._enc_ff_swish = True
        self._enc_ff_no_bias = True
        # Encoder Attention Configs
        self._enc_attn_flash = enc_attn_flash
        self._enc_attn_one_kv_head = False
        # Encoder Layer Normalization Configs
        self._enc_use_simple_rmsnorm = True
        # Decoder Feed Forward Configs
        self._dec_ff_glu = True
        self._dec_ff_swish = True
        self._dec_ff_no_bias = True
        # Decoder Positional Encoding Configs
        self._dec_rel_pos_bias = False
        self._dec_rotary_pos_emb = True
        self._dec_rotary_xpos = False
        # Decoder Attention Configs
        self._dec_attn_flash = dec_attn_flash
        self._dec_attn_one_kv_head = False
        self._dec_gate_residual = False
        self._dec_cross_residual_attn = False
        # Decoder Layer Normalization Configs
        self._dec_use_simple_rmsnorm = True
        self._dec_sandwich_norm = False
        # Model Configs
        self._use_flash = use_flash
        self._activation = activation
        self._positional_encoding_type = positional_encoding_type
        self._tie_token_embeddings = True
        self._tokenizer2_path = None

    def check_constraints(self):
        if self._feedforward_size == 0:
            self._feedforward_size = self._embedding_size * 2
        if self._hidden_size == 0:
            self._hidden_size = self._embedding_size

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
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def enc_vocab_size(self):
        return self._enc_vocab_size

    @property
    def enc_layers(self):
        return self._enc_layers

    @property
    def source_max_len(self):
        return self._source_max_len

    @property
    def dec_vocab_size(self):
        return self._dec_vocab_size

    @property
    def dec_layers(self):
        return self._dec_layers

    @property
    def target_max_len(self):
        return self._target_max_len

    @property
    def enc_dropout(self):
        return self._enc_dropout

    @property
    def enc_layer_dropout(self):
        return self._enc_layer_dropout

    @property
    def enc_attn_dropout(self):
        return self._enc_attn_dropout

    @property
    def enc_ff_dropout(self):
        return self._enc_ff_dropout

    @property
    def dec_dropout(self):
        return self._dec_dropout

    @property
    def dec_layer_dropout(self):
        return self._dec_layer_dropout

    @property
    def dec_attn_dropout(self):
        return self._dec_attn_dropout

    @property
    def dec_ff_dropout(self):
        return self._dec_ff_dropout

    @property
    def enc_ff_glu(self):
        return self._enc_ff_glu

    @property
    def enc_ff_swish(self):
        return self._enc_ff_swish

    @property
    def enc_ff_no_bias(self):
        return self._enc_ff_no_bias

    @property
    def enc_attn_flash(self):
        return self._enc_attn_flash

    @property
    def enc_attn_one_kv_head(self):
        return self._enc_attn_one_kv_head

    @property
    def enc_use_simple_rmsnorm(self):
        return self._enc_use_simple_rmsnorm

    @property
    def dec_ff_glu(self):
        return self._dec_ff_glu

    @property
    def dec_ff_swish(self):
        return self._dec_ff_swish

    @property
    def dec_ff_no_bias(self):
        return self._dec_ff_no_bias

    @property
    def dec_rel_pos_bias(self):
        return self._dec_rel_pos_bias

    @property
    def dec_rotary_pos_emb(self):
        return self._dec_rotary_pos_emb

    @property
    def dec_rotary_xpos(self):
        return self._dec_rotary_xpos

    @property
    def dec_attn_flash(self):
        return self._dec_attn_flash

    @property
    def dec_attn_one_kv_head(self):
        return self._dec_attn_one_kv_head

    @property
    def dec_gate_residual(self):
        return self._dec_gate_residual

    @property
    def dec_cross_residual_attn(self):
        return self._dec_cross_residual_attn

    @property
    def dec_use_simple_rmsnorm(self):
        return self._dec_use_simple_rmsnorm

    @property
    def dec_sandwich_norm(self):
        return self._dec_sandwich_norm

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
    def tie_token_embeddings(self):
        return self._tie_token_embeddings

    @property
    def tokenizer2_path(self):
        return self._tokenizer2_path

    @tokenizer2_path.setter
    def tokenizer2_path(self, value):
        assert value is None or isinstance(value, str | Path), "Tokenizer path can be None or a string or pathlib.Path object"
        self._tokenizer2_path = value

    def get_configurable_parameters(self, force_cpu: bool = False) -> dict[str, list[Any]]:
        config = deepcopy(configs.XTR_PARAMETERS)
        if torch.cuda.is_available() and not force_cpu and not self.cross_platform_compatibility:
            config["optimizer"].extend([f"{opt}_8bit" for opt in config["optimizer"]])
        return config


########################################################################################################################
#                                                Encoder-Decoder Model                                                 #
########################################################################################################################
class EncoderDecoderModel(CustomModel):
    """Encoder-Decoder custom transformer model."""

    def __init__(self, config: EncoderDecoderConfig, tokenizer: BaseTokenizer,
                 output_tokenizer: Optional[BaseTokenizer] = None, logger: Optional[Logger] = None, force_cpu: bool = False):
        assert isinstance(config, EncoderDecoderConfig), "config must be an instance of EncoderDecoderConfig"
        config.check_constraints()
        flash_runtime_supported = torch.cuda.is_available() and not force_cpu and not getattr(config, "force_cpu", False)
        if config.enc_vocab_size > config.dec_vocab_size:
            encoder_vocab_size = config.enc_vocab_size
            decoder_vocab_size = config.enc_vocab_size
        else:
            encoder_vocab_size = config.dec_vocab_size
            decoder_vocab_size = config.dec_vocab_size
        model = XTransformer(
            dim=config.embedding_size,
            tie_token_emb=config.tie_token_embeddings,
            dec_cross_residual_attn=config.dec_cross_residual_attn,
            # Encoder Configs
            enc_num_tokens=encoder_vocab_size,
            enc_depth=config.enc_layers,
            enc_heads=config.num_heads,
            enc_max_seq_len=config.source_max_len,
            enc_emb_dropout=config.enc_dropout,
            enc_layer_dropout=config.enc_layer_dropout,
            enc_attn_dropout=config.enc_attn_dropout,
            enc_ff_dropout=config.enc_ff_dropout,
            enc_ff_glu=config.enc_ff_glu,
            enc_ff_swish=config.enc_ff_swish,
            enc_ff_no_bias=config.enc_ff_no_bias,
            enc_attn_flash=config.enc_attn_flash and flash_runtime_supported,
            enc_use_simple_rmsnorm=config.enc_use_simple_rmsnorm,
            enc_attn_one_kv_head=config.enc_attn_one_kv_head,
            # Decoder Configs
            dec_num_tokens=decoder_vocab_size,
            dec_depth=config.dec_layers,
            dec_heads=config.num_heads,
            dec_max_seq_len=config.target_max_len,
            dec_emb_dropout=config.dec_dropout,
            dec_attn_dropout=config.dec_attn_dropout,
            dec_layer_dropout=config.dec_layer_dropout,
            dec_ff_dropout=config.dec_ff_dropout,
            dec_ff_glu=config.dec_ff_glu,
            dec_ff_swish=config.dec_ff_swish,
            dec_ff_no_bias=config.dec_ff_no_bias,
            dec_attn_flash=config.dec_attn_flash and flash_runtime_supported,
            dec_use_simple_rmsnorm=config.dec_use_simple_rmsnorm,
            dec_sandwich_norm=config.dec_sandwich_norm,
            dec_attn_one_kv_head=config.dec_attn_one_kv_head,
            dec_gate_residual=config.dec_gate_residual,
            dec_rel_pos_bias=config.dec_rel_pos_bias,
            dec_rotary_pos_emb=config.dec_rotary_pos_emb,
            dec_rotary_xpos=config.dec_rotary_xpos,
        )

        super().__init__(model, config, tokenizer, output_tokenizer, logger, force_cpu)

        if self._output_tokenizer is not None:
            active_tokenizer = self._output_tokenizer
            self._valid_output_ids = torch.tensor(self._output_tokenizer.token_ids, device=self.device, dtype=torch.long)
        else:
            active_tokenizer =  self._tokenizer
            self._valid_output_ids = None

        if active_tokenizer is None:
            raise ValueError("At least one tokenizer must be provided to initialize TwoPhaseModel.")

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=active_tokenizer.pad_token_id)
        self.__bos_token_id__ = active_tokenizer.token_to_id(active_tokenizer.bos_token)
        self.__eos_token_id__ = active_tokenizer.token_to_id(active_tokenizer.eos_token)
        assert active_tokenizer.pad_token_id is not None, "Pad token ID must be provided. Tokenizer is not properly initialized."
        assert self.__bos_token_id__ is not None, "BOS token ID must be provided. Tokenizer is not properly initialized."
        assert self.__eos_token_id__ is not None, "EOS token ID must be provided. Tokenizer is not properly initialized."

    def check_tokenizers(self, *, dataframe: dataset.DataFrame, **kwargs):
        input_tokenizer_path = kwargs.get("input_tokenizer_path", self.config.tokenizer_path)
        output_tokenizer_path = kwargs.get("output_tokenizer_path", self.config.tokenizer2_path)
        self.__check_both_tokenizers__(input_tokenizer_path, output_tokenizer_path, dataframe)

    def __forge_dataset__(self, dataframe: dataset.DataFrame) -> dataset.TorchDataset:
        """
        This method is responsible for creating a TorchDataset from the given DataFrame. Do not call this method directly.
        :param dataframe: pandas DataFrame object
        :return: TorchSeq2SeqDataset object
        """
        return dataset.TorchSeq2SeqDataset(dataframe, self.config.source, self.config.target, self._tokenizer, self._output_tokenizer, self.config.instructions)


    def __forward__(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This method is responsible for passing the input through the model and returning the loss.
        Do not call this method directly. It is called by train, test methods.
        :param sample: dict[str, torch.Tensor] containing input_ids, output_ids, attention_mask (input padding mask)
        """
        input_ids = sample["input_ids"]
        output_ids = sample["output_ids"]
        attention_mask = sample["attention_mask"].bool()
        return self._model(input_ids, output_ids, mask=attention_mask)

    @torch.inference_mode()
    def generate(self, input_data: Any, max_length: int = 256, **_) -> Any:
        """
        Generate output from the input data.
        :param input_data: Vector of input data or a single string
        :param max_length: Maximum length of the output
        :return: Generated output as a string or a vector
        """
        tokenized_input = self._tokenizer(input_data)
        input_ids = tokenized_input["input_ids"].unsqueeze(0)
        attention_mask = tokenized_input["attention_mask"].bool().unsqueeze(0)
        tgt_ids = torch.full((1, 1), self.__bos_token_id__, device=self.device, dtype=torch.long)
        output = self._model.generate(input_ids, tgt_ids, max_length, mask=attention_mask)
        if self._valid_output_ids is not None:
            mask = ~torch.isin(output, self._valid_output_ids)
            output = output.masked_fill(mask, self.__bos_token_id__)
        return self._output_tokenizer.reconstruct(output)

    def save_model(self, filename: str = "model.pth"):
        if self.epochwise_checkpointing_enabled:
            filename = f"checkpoint_epoch{self.config.epoch}.pth"
        if self._output_tokenizer != self._tokenizer and self._output_tokenizer is not None:
            tokenizer_path = self.config.output_dir.joinpath("tokenizer2")
            self._output_tokenizer.save(tokenizer_path)
            self.config.tokenizer2_path = tokenizer_path.relative_to(self.config.output_dir)
        else:
            self.config.tokenizer2_path = None
        super().save_model(filename)

    @staticmethod
    def __init_components__(filepath: str | Path, logger: Optional[Logger] = None):
        filepath = CustomModel.check_model_filepath(filepath, logger=logger)
        config = CustomConfig.from_pretrained(filepath, logger)
        if config.tokenizer_path is not None:
            tokenizer = load_module(config.output_dir.joinpath(config.tokenizer_path), logger)
        else:
            tokenizer = None
        if config.tokenizer2_path is not None:
            output_tokenizer = load_module(config.output_dir.joinpath(config.tokenizer2_path), logger)
        else:
            output_tokenizer = None
        return config, tokenizer, output_tokenizer

    @staticmethod
    def from_pretrained(filepath: str | Path, logger: Optional[Logger] = None) -> "EncoderDecoderModel":
        config, tokenizer, output_tokenizer = EncoderDecoderModel.__init_components__(filepath, logger)
        return EncoderDecoderModel(config, tokenizer, output_tokenizer, logger)
