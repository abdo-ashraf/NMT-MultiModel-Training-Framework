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

        self.num_train_epochs = config.get("num_train_epochs")
        assert isinstance(self.num_train_epochs, int), "num_train_epochs must be an integer."

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

        self.maxlen = config.get("maxlen")
        assert isinstance(self.maxlen, int), "maxlen must be an integer."

        self.onnx = config.get("onnx")
        assert isinstance(self.onnx, bool), "onnx must be a boolean."

        self.run_name = config.get("run_name")
        assert isinstance(self.run_name, str), "run_name must be a string."

        self.pin_memory = config.get("pin_memory")
        assert isinstance(self.pin_memory, bool), "pin_memory must be a boolean."

        self.warmup_epochs = config.get("warmup_epochs")
        assert isinstance(self.warmup_epochs, int), "warmup_epochs must be an integer."

        self.save_epochs = config.get("save_epochs")
        assert isinstance(self.save_epochs, int), "save_epochs must be an integer."

        self.torch_compile = config.get("torch_compile")
        assert isinstance(self.torch_compile, bool), "torch_compile must be a boolean."


    def __repr__(self):
        return (f"TrainingArguments(output_dir='{self.output_dir}', learning_rate={self.learning_rate}, "
                f"num_train_epochs={self.num_train_epochs}, seed={self.seed}, precision='{self.precision}', "
                f"device='{self.device}', batch_size={self.batch_size}, cpu_num_workers={self.cpu_num_workers}, "
                f"weight_decay={self.weight_decay}, maxlen={self.maxlen}, onnx={self.onnx}, run_name='{self.run_name}', "
                f"pin_memory={self.pin_memory}, warmup_epochs={self.warmup_epochs}, save_epochs={self.save_epochs}, "
                f"torch_compile={self.torch_compile})")
