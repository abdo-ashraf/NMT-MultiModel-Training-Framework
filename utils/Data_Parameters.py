## Data params: train_csv_path, valid_csv_path, batch_size, num_workers, seed, device, out_dir, maxlen
## Tokenizer params: src_vocab_size, trg_vocab_size, out_dir, lang1_model_prefix, lang2_model_prefix, lang1_character_coverage, lang2_character_coverage
## Model params: model_type, epochs, dim_embed, dim_model, dim_feedforward, num_layers, dropout, learning_rate, weight_decay, out_dir

import os
from torch import Generator

class DataParams():
    def __init__(self, train_csv_path:str, valid_csv_path:str,
                 epochs:int, batch_size:int, num_workers:int,
                 seed:int, device:str, out_dir:str, maxlen:int):

        assert os.path.exists(train_csv_path), f"Path: {train_csv_path} doesn't exists"
        self.train_csv_path = train_csv_path

        assert os.path.exists(valid_csv_path), f"Path: {valid_csv_path} doesn't exists"
        self.valid_csv_path = valid_csv_path

        assert isinstance(epochs, int), f"epochs must be Integer"
        self.epochs = epochs

        assert isinstance(batch_size, int), f"Batch_size must be Integer"
        self.batch_size = batch_size

        assert isinstance(num_workers, int), f"num_workers must be Integer"
        self.num_workers = num_workers

        assert isinstance(seed, int), f"seed must be Integer"
        self.generator = Generator().manual_seed(seed)

        assert device in ['cuda', 'cpu'], f"device should be 'cuda' or 'cpu'"
        self.device = device

        assert isinstance(out_dir, str), f"out_dir must be String"
        self.plots_dir = os.path.join(out_dir, 'plots')
        if not os.path.exists(self.plots_dir):
            print(f"{self.plots_dir} does not exists")
            print(f'Making dirs tree @{self.plots_dir}...')
            os.makedirs(self.plots_dir, exist_ok=True)
            print('Done.')

        assert isinstance(maxlen, int), f"maxlen must be Integer"
        self.maxlen = maxlen
