from enum import Enum
from pathlib import Path

from wandb import login as wandb_login, finish as wandb_finish
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset
from trl import SFTTrainer, SFTConfig
from pandas import read_csv, read_json, DataFrame
from torch import save as torch_save, load as torch_load

try:
    from unsloth import unsloth_train
except ImportError:
    def unsloth_train(trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)

SEED = 31921

TEMPLATES = {
    "llama2": "<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_input} [/INST] {model_output} </s>",
    "gemma1":  "<start_of_turn>user\n{system_msg}\n\n{user_input} <end_of_turn>\n<start_of_turn>model\n{model_output} <end_of_turn>",
    "llama3": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{model_output}<|eot_id|>",
    "phi3": "<|system|>\n{system_msg}<|end|>\n<|user|>\n{user_input}<|end|>\n<|assistant|>\n{model_output}<|end|>",
    "qwen2": "<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n{model_output}<|im_end|>"
}

QUERY_TEMPLATES = {
    "llama2": "<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_input} [/INST]",
    "gemma1":  "<start_of_turn>user\n{system_msg}\n\n{user_input} <end_of_turn>\n<start_of_turn>model\n",
    "llama3": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>",
    "phi3": "<|system|>\n{system_msg}<|end|>\n<|user|>\n{user_input}<|end|>\n<|assistant|>",
    "qwen2": "<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
}

STOP_TOKEN_SEQUENCE = {
    "llama2": "[/INST]",
    "gemma1": "<start_of_turn>model",
    "llama3": "assistant<|end_header_id|>",
    "phi3": "<|assistant|>",
    "qwen2": "<|im_start|>assistant"
}

TEMPLATE_MAPPING = {
    "unsloth/codellama-7b-bnb-4bit": "llama2",
    "unsloth/codegemma-7b-it-bnb-4bit": "gemma1",
    "unsloth/gemma-2-2b-it-bnb-4bit": "gemma1",
    "unsloth/gemma-2-9b-it-bnb-4bit": "gemma1",
    "unsloth/gemma-2-27b-bnb-4bit": "gemma1",
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit": "llama3",
    "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit": "llama3",
    "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit": "llama3",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit": "llama3",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit": "llama3",
    "unsloth/Phi-3.5-mini-instruct-bnb-4bit": "phi3",
    "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit": "qwen2",
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit": "qwen2",
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit": "qwen2",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit": "qwen2",
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": "qwen2",
    "unsloth/Qwen2.5-Coder-0.5B-bnb-4bit": "qwen2",
    "unsloth/Qwen2.5-Coder-1.5B-bnb-4bit": "qwen2",
    "unsloth/Qwen2.5-Coder-3B-bnb-4bit": "qwen2",
    "unsloth/Qwen2.5-Coder-7B-bnb-4bit": "qwen2",
    "unsloth/Qwen2.5-Coder-32B-bnb-4bit": "qwen2",
}


class ModelType(Enum):
    UNKNOWN = "unknown"
    CodeLlama = "unsloth/codellama-7b-bnb-4bit"
    CodeGemma = "unsloth/codegemma-7b-it-bnb-4bit"
    Gemma2_Small = "unsloth/gemma-2-2b-it-bnb-4bit"
    Gemma2_Medium = "unsloth/gemma-2-9b-it-bnb-4bit"
    Gemma2_Large = "unsloth/gemma-2-27b-bnb-4bit"
    Llama31_Medium = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    Llama31_Large = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    Llama31_XLarge = "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"
    Llama32_Micro = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    Llama32_Small = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    Phi35_Mini = "unsloth/Phi-3.5-mini-instruct-bnb-4bit"
    Qwen25_Micro = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
    Qwen25_Small = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    Qwen25_Medium = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    Qwen25_Large = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    Qwen25_XLarge = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
    Qwen25Coder_Micro = "unsloth/Qwen2.5-Coder-0.5B-bnb-4bit"
    Qwen25Coder_Small = "unsloth/Qwen2.5-Coder-1.5B-bnb-4bit"
    Qwen25Coder_Medium = "unsloth/Qwen2.5-Coder-3B-bnb-4bit"
    Qwen25Coder_Large = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"
    Qwen25Coder_XLarge = "unsloth/Qwen2.5-Coder-32B-bnb-4bit"


    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN


INSTRUCTION_TEMPLATE = "You are a converter program that translates {source} to {target}. Translate the following data:"


class Model:
    def __init__(self, model_type: str | ModelType,
                 source: str,
                 target: str,
                 output_dir: str | Path,
                 *, hf_token: str | None = None,
                 model_instruction: str | None = None, **_):
        self.model = None
        self.tokenizer = None
        self.hf_token = hf_token
        model_name = model_type.value if isinstance(model_type, ModelType) else model_type
        name = f"{model_name.replace('unsloth/', '')}_{source}_{target}"
        output_dir = Path(output_dir).resolve()
        self.test_dir = output_dir.joinpath("test")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        template_type = TEMPLATE_MAPPING[model_name]
        self.test_prompt_template = QUERY_TEMPLATES[template_type]
        self.stop_sequence = STOP_TOKEN_SEQUENCE[template_type]
        print(f"Model type: {model_name} | Template type: {template_type}")

        self.config = {
            "model_name": model_name,
            "name": name,
            "source": source,
            "target": target,
            "source_column": source.lower(),
            "target_column": target.lower(),
            "max_sequence_length": 0,
            "model_instruction": model_instruction.format(source=source, target=target) if model_instruction is not None else None,
        }

    def init_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config["model_name"],
            max_seq_length=self.config["max_sequence_length"],
            token=self.hf_token,
            load_in_4bit=True,
            trust_remote_code=True,
        )

        self.model = FastLanguageModel.get_peft_model(
            model=self.model,
            r=64,
            lora_alpha=32,
            use_gradient_checkpointing=True,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
        )
        FastLanguageModel.for_inference(self.model)

    def test(self, filepath: str | Path, *, output_dir: str | Path | None = None, max_new_tokens: int = 256, test_name: str | None = None) -> DataFrame:
        assert self.model is not None and self.tokenizer is not None, "Model is not initialized. Please train or load the model first."
        if test_name is None:
            test_name = ""
        else:
            test_name += "_"
        if output_dir is None:
            output_dir = self.test_dir
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)
        else:
            raise TypeError("Invalid output directory type. Must be a string or Path object.")
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.load_data(filepath, self.config["source_column"], self.config["target_column"])
        formatted_data = self.format_input_output_data(df, self.config["source_column"], self.config["target_column"])
        results = []
        for formatted_input, original_input, expected_output in formatted_data:
            input_ids = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**input_ids, max_new_tokens=max_new_tokens, use_cache=True)
            outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            stop_pos = outputs.find(self.stop_sequence)
            model_output = outputs[stop_pos+len(self.stop_sequence):].strip() if stop_pos != -1 else outputs
            results.append((original_input, expected_output, model_output))
        output_dir = output_dir.joinpath(f"{test_name}TestResults_{self.config['name']}.csv").as_posix()
        df = DataFrame(results, columns=["Input", "Expected Output", "Model Output"])
        df.to_csv(output_dir, index=False)
        return df

    def prompt_model(self, input_data, *, max_new_tokens: int = 256) -> str:
        assert self.model is not None and self.tokenizer is not None, "Model is not initialized. Please train or load the model first."
        if '{system_msg}' in self.test_prompt_template:
            assert self.config["model_instruction"] is not None, "Selected templates requires the specification of instructions / system message."
            formatted_input = self.test_prompt_template.format(system_msg=self.config["model_instruction"], user_input=input_data)
        else:
            formatted_input = self.test_prompt_template.format(user_input=input_data)
        inputs_ids = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs_ids, max_new_tokens=max_new_tokens, use_cache=True)
        outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        stop_pos = outputs.find(self.stop_sequence)
        return outputs[stop_pos+len(self.stop_sequence):].strip() if stop_pos != -1 else outputs

    def format_input_output_data(self, df: DataFrame, source: str, target: str | None = None) -> list[tuple]:
        assert isinstance(df, DataFrame), "Input data df must be a pandas DataFrame"
        assert source in df.columns, f"Source column {source} not found in the input data."
        source_values = df[source].values.tolist()
        if '{system_msg}' in self.test_prompt_template:
            assert self.config["model_instruction"] is not None, "Selected templates requires the specification of instructions / system message."
            formatted_source = [self.test_prompt_template.format(system_msg=self.config["model_instruction"], user_input=src) for src in source_values]
        else:
            formatted_source = [self.test_prompt_template.format(user_input=src) for src in source_values]
        if target is None:
            return list(zip(formatted_source, source_values))
        else:
            target_values = df[target].values.tolist()
            return list(zip(formatted_source, source_values, target_values))

    @staticmethod
    def load_data(filepath: str | Path, source: str, target: str) -> DataFrame | None:
        assert isinstance(filepath, str | Path), "Filepath must be a string or Path object."
        filepath = Path(filepath)
        assert filepath.exists(), f"File not found: {filepath}"
        assert filepath.is_file(), f"Filepath point to a directory: {filepath}"
        if filepath.suffix == ".csv":
            train_dataframe = read_csv(filepath)
        elif filepath.suffix == ".json":
            train_dataframe = read_json(filepath)
        else:
            raise ValueError(f"Invalid file format: {filepath}\n Only .csv and .json files are supported.")
        assert source in train_dataframe.columns, f"Source column {source} not found in the input data."
        assert target in train_dataframe.columns, f"Target column {target} not found in the input data."
        return train_dataframe


class FineTunedModel(Model):
    def __init__(self, model_type: str | ModelType,
                 source: str,
                 target: str,
                 output_dir: str | Path,
                 *, hf_token: str | None = None,
                 wandb_token: str | None = None,
                 model_instruction: str | None = None, **_):
        super().__init__(model_type, source, target, output_dir, hf_token=hf_token, model_instruction=model_instruction)
        output_dir = Path(output_dir).resolve()
        self.output_dir = output_dir.joinpath("model")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_name = model_type.value if isinstance(model_type, ModelType) else model_type
        template_type = TEMPLATE_MAPPING[model_name]
        self.train_prompt_template = TEMPLATES[template_type]
        self.wandb_token = wandb_token

    def train(self, filepath: str | Path,
              num_train_epochs: int, *,
              batch_size: int = 4,
              gradient_accumulation_steps: int = 4,
              warmup_steps: int = 5,
              learning_rate: float = 2e-4,
              weight_decay: float = 0.01,
              logging_steps: int = 100,
              save_total_limit: int = 1,
              seed: int = SEED):

        if self.wandb_token is not None:
            wandb_login(key=self.wandb_token)
        dataframe = self.load_data(filepath, self.config["source_column"], self.config["target_column"])
        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"], token=self.hf_token, trust_remote_code=True, clean_up_tokenization_spaces=False)
        max_train_input_length = max(len(tokens) for tokens in tokenizer(dataframe[self.config["source_column"]].values.tolist(), return_tensors=None, padding=False, truncation=False)["input_ids"])
        max_train_output_length = max(len(tokens) for tokens in tokenizer(dataframe[self.config["target_column"]].values.tolist(), return_tensors=None, padding=False, truncation=False)["input_ids"])
        instruction_len = len(self.config["model_instruction"]) if self.config["model_instruction"] is not None else 0
        self.config["max_sequence_length"] = max_train_input_length + max_train_output_length + instruction_len + 10

        if self.model is None:
            self.init_model()

        FastLanguageModel.for_training(self.model)

        dataset = self.create_formatted_dataset(dataframe, self.config["source_column"], self.config["target_column"], eos_token=self.tokenizer.eos_token)

        train_config = SFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=logging_steps,
            optim="adamw_8bit",
            weight_decay=weight_decay,
            lr_scheduler_type="linear",
            seed=seed,
            output_dir=self.output_dir.as_posix(),
            report_to="wandb",
            save_total_limit=save_total_limit,
            run_name=self.config["name"],
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.config["max_sequence_length"],
            dataset_num_proc=8,
            packing=False,  # Can make training 5x faster for short sequences.
            args=train_config
        )

        try:
            unsloth_train(trainer)
        except Exception:
            trainer.train()

        if self.wandb_token is not None:
            wandb_finish()

        self.save_model()
        FastLanguageModel.for_inference(self.model)

    def save_model(self):
        assert self.model is not None and self.tokenizer is not None, "Model is not initialized. Please train or load the model first."
        self.model.save_pretrained(self.output_dir.as_posix())
        self.tokenizer.save_pretrained(self.output_dir.as_posix())
        self.config["checkpoint_path"] = self.output_dir.as_posix()
        torch_save(self.config, self.output_dir.joinpath("config.pt").as_posix())

    @staticmethod
    def load_model(filepath: str | Path):
        assert isinstance(filepath, str | Path), "Filepath must be a string or Path object."
        filepath = Path(filepath)
        assert filepath.exists(), f"File not found: {filepath}"
        if filepath.is_dir():
            candidate = filepath.joinpath("config.pt")
            if candidate.is_file():
                filepath = candidate
            else:
                candidate = filepath / "data/model/config.pt"
                if candidate.is_file():
                    filepath = candidate
                else:
                    raise FileNotFoundError("Config file not found in the specified directory.")
        else:
            assert filepath.suffix == ".pt", f"Invalid file format: {filepath}"
        config = torch_load(filepath)
        assert isinstance(config, dict), "Invalid config file."
        if not Path(config["checkpoint_path"]).exists():
            config["checkpoint_path"] = filepath.parent.as_posix()
        config.setdefault("output_dir", filepath.parent.parent.as_posix())
        config.setdefault("model_type", config["model_name"])
        ft_model = FineTunedModel(**config)
        ft_model.model, ft_model.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["checkpoint_path"],
            load_in_4bit=True,
            trust_remote_code=True,
            max_seq_length=config["max_sequence_length"],
        )
        FastLanguageModel.for_inference(ft_model.model)
        return ft_model

    def create_formatted_dataset(self, df: DataFrame, source: str, target: str | list[str], *, eos_token: str | None) -> HFDataset:
        assert isinstance(df, DataFrame), "Input data df must be a pandas DataFrame"
        assert source in df.columns, f"Source column {source} not found in the input data."
        assert target in df.columns, f"Target column {target} not found in the input data."
        if eos_token is None: eos_token = ""

        source_values = df[source].values.tolist()
        if isinstance(target, list):
            target_values = df[target].astype(str).agg(' '.join, axis=1).tolist()
        else:
            target_values = df[target].values.tolist()
        if '{system_msg}' in self.train_prompt_template:
            assert self.config["model_instruction"] is not None, "Selected templates requires the specification of instructions / system message."
            formatted_data = [self.train_prompt_template.format(system_msg=self.config["model_instruction"], user_input=src, model_output=trg) + eos_token for
                              src, trg in zip(source_values, target_values)]
        else:
            formatted_data = [self.train_prompt_template.format(user_input=src, model_output=trg) + eos_token for src, trg in
                              zip(source_values, target_values)]
        print(formatted_data[0])
        return HFDataset.from_dict({"text": formatted_data})
