import os
import json

class TrainingArguments:
    """
    A class to parse and manage training configuration parameters from a JSON file.
    """
    def __init__(self, config_json_path: str):
        """
        Initialize TrainingArguments from a configuration JSON file.

        Args:
            config_json_path (str): Path to the JSON configuration file.

        Raises:
            AssertionError: If any configuration value is invalid or missing.
        """
        assert os.path.exists(config_json_path), f"{config_json_path} : configuration file not found."

        # Load configuration from JSON file
        with open(config_json_path, 'r') as file:
            config = json.load(file)

        # Validate and assign attributes
        self.save_models_dir = config.get("save_models_dir")
        assert os.path.exists(self.save_models_dir), f"{self.save_models_dir} : output directory not found."

        self.save_plots_dir = config.get("save_plots_dir")
        assert os.path.exists(self.save_plots_dir), f"{self.save_plots_dir} : Plot directory not found."

        self.learning_rate = config.get("learning_rate")
        assert isinstance(self.learning_rate, float), "learning_rate must be a float."

        self.max_steps = config.get("max_steps")
        assert isinstance(self.max_steps, int), "max_steps must be an integer."

        self.seed = config.get("seed")
        assert isinstance(self.seed, int), "seed must be an integer."

        self.precision = config.get("precision")
        assert self.precision in ["high", "highest"], "Precision must be one of ['high', 'highest']."

        self.device = config.get("device")
        assert isinstance(self.device, str), "device must be a string."

        self.batch_size = config.get("batch_size")
        assert isinstance(self.batch_size, int), "batch_size must be an integer."

        self.cpu_num_workers = config.get("cpu_num_workers")
        assert isinstance(self.cpu_num_workers, int), "cpu_num_workers must be an integer."

        self.weight_decay = config.get("weight_decay")
        assert isinstance(self.weight_decay, float), "weight_decay must be a float."

        self.onnx = config.get("onnx")
        assert isinstance(self.onnx, bool), "onnx must be a boolean."

        self.run_name = config.get("run_name")
        assert isinstance(self.run_name, str), "run_name must be a string."

        self.pin_memory = config.get("pin_memory")
        assert isinstance(self.pin_memory, bool), "pin_memory must be a boolean."

        self.warmup_steps = config.get("warmup_steps")
        assert isinstance(self.warmup_steps, int), "warmup_steps must be an integer."

        self.save_steps = config.get("save_steps")
        assert isinstance(self.save_steps, int), "save_steps must be an integer."

        self.eval_steps = config.get("eval_steps")
        assert isinstance(self.eval_steps, int), "eval_steps must be an integer."

        self.torch_compile = config.get("torch_compile")
        assert isinstance(self.torch_compile, bool), "torch_compile must be a boolean."

        self.lr_decay_ratio = config.get("lr_decay_ratio")
        assert isinstance(self.lr_decay_ratio, float), "lr_decay_ratio must be a float."


    def __repr__(self):
        """
        String representation of the TrainingArguments object.

        Returns:
            str: A string showing all attributes and their values.
        """
        return ("TrainingArguments(\n" +
                f"  save_models_dir='{self.save_models_dir}',\n" +
                f"  save_plots_dir='{self.save_plots_dir}',\n" +
                f"  learning_rate={self.learning_rate},\n" +
                f"  max_steps={self.max_steps},\n" +
                f"  seed={self.seed},\n" +
                f"  precision='{self.precision}',\n" +
                f"  device='{self.device}',\n" +
                f"  batch_size={self.batch_size},\n" +
                f"  cpu_num_workers={self.cpu_num_workers},\n" +
                f"  weight_decay={self.weight_decay},\n" +
                f"  onnx={self.onnx},\n" +
                f"  run_name='{self.run_name}',\n" +
                f"  pin_memory={self.pin_memory},\n" +
                f"  warmup_steps={self.warmup_steps},\n" +
                f"  save_steps={self.save_steps},\n" +
                f"  eval_steps={self.eval_steps},\n" +
                f"  torch_compile={self.torch_compile}\n" +
                f"  lr_decay_ratio={self.lr_decay_ratio}\n" +
                ")")
