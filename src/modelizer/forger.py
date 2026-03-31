import inspect
import importlib

from typing import Optional
from modelizer.utils import Logger
from modelizer.tokenizers.abstract import BaseTokenizer
from modelizer.models.abstract import BaseModel, BaseConfig


def find_model_class(config: BaseConfig, class_name: Optional[str | type] = None) -> type | None:
    """Finds the model class for a given config using registration, naming convention, or module search."""
    if isinstance(class_name, type):
        return class_name

    if hasattr(BaseModel, "get_registered_model"):
        lookup_name = class_name if class_name else config.__class__.__name__
        candidate = BaseModel.get_registered_model(lookup_name)
        if isinstance(candidate, type):
            return candidate

    if class_name is None:
        if config.__class__.__name__.endswith("Config"):
            class_name = config.__class__.__name__[:-6] + "Model"
        else:
            class_name = config.__class__.__name__ + "Model"

    config_module = inspect.getmodule(config.__class__)

    if config_module and hasattr(config_module, class_name):
        candidate = getattr(config_module, class_name)
        if isinstance(candidate, type):
            return candidate

    module_patterns: list[str] = []
    if config_module:
        module_patterns.append(f"{config_module.__name__}.model")
        module_patterns.append(f"{config_module.__name__}.models")

        if '.' in config_module.__name__:
            parent_mod = '.'.join(config_module.__name__.split('.')[:-1])
            module_patterns.append(f"{parent_mod}.model")
            module_patterns.append(f"{parent_mod}.models")

    for mod_name in module_patterns:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, class_name):
            return getattr(mod, class_name)

    return None


def forge_custom_model(config: BaseConfig,
                       tokenizer: Optional[BaseTokenizer],
                       output_tokenizer: Optional[BaseTokenizer],
                       logger: Optional[Logger],
                       force_cpu: bool = False) -> BaseModel:
    """Factory for arbitrary BaseModel subclasses. Tries explicit registration, then naming convention, then module search."""
    logger = Logger.forge(logger)
    config.force_cpu = bool(config.force_cpu or force_cpu)

    model_class = None
    predefined = getattr(config, "model_class", None)

    if isinstance(predefined, type):
        model_class = predefined
    elif predefined is not None:
        model_class = find_model_class(config, predefined)

    if model_class is None:
        model_class = find_model_class(config)

    if model_class is None:
        raise ValueError(f"Could not find model class for config {config.__class__.__name__}."
                         " Use naming convention or register explicitly.")
    elif not issubclass(model_class, BaseModel):
        raise TypeError(f"{model_class.__name__} must extend {BaseModel.__name__}")

    signature = inspect.signature(model_class.__init__)
    params = list(signature.parameters.keys())[1:]  # skip self
    kwargs = {'config': config, 'logger': logger}

    if 'tokenizer' in params and tokenizer is not None:
        kwargs['tokenizer'] = tokenizer
    if 'output_tokenizer' in params and output_tokenizer is not None:
        kwargs['output_tokenizer'] = output_tokenizer
    assert model_class is not None, "model_class must not be None at this point"
    # noinspection PyCallingNonCallable
    return model_class(**kwargs)
