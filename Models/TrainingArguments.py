class TrainingArguments():
    def __init__(self, output_dir, learning_rate, num_train_epochs,
                 seed, precision, device, plots_dir, optimizer,
                 batch_size, cpu_num_workers, weight_decay, maxlen,
                 onnx, run_name, pin_memory):
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        # self.save_steps = save_steps
        self.seed = seed
        self.precision = precision
        self.device = device
        self.plots_dir = plots_dir
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.cpu_num_workers = cpu_num_workers
        self.weight_decay = weight_decay
        self.maxlen = maxlen
        self.onnx = onnx
        self.run_name = run_name
        self.pin_memory = pin_memory
