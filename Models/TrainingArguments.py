class TrainingArguments():
    def __init__(self, output_dir, learning_rate, num_train_epochs,
                 seed, precision, device, plots_dir, optimizer,
                 batch_size, cpu_num_workers, weight_decay, maxlen,
                 onnx, run_name, pin_memory, warmup_steps, save_epochs,
                 torch_compile, experiment_name_mlflow, flash_attention):
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.save_epochs = save_epochs
        self.seed = seed
        self.precision = precision ## "float32" or "float16" or "TensorFloat32" or "BFloat16"
        self.torch_compile = torch_compile
        self.device = device
        self.plots_dir = plots_dir
        self.optimizer = optimizer
        self.cpu_num_workers = cpu_num_workers
        self.weight_decay = weight_decay
        self.maxlen = maxlen
        self.onnx = onnx
        self.run_name = run_name
        self.pin_memory = pin_memory
        self.experiment_name_mlflow = experiment_name_mlflow
        self.flash_attention = flash_attention
