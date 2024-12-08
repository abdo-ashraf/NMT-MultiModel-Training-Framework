## Data params: train_csv_path, valid_csv_path, batch_size, num_workers, seed, device, out_dir, maxlen
## Tokenizer params: src_vocab_size, trg_vocab_size, out_dir, lang1_model_prefix, lang2_model_prefix, lang1_character_coverage, lang2_character_coverage
## Model params: model_type, epochs, dim_embed, dim_model, dim_feedforward, num_layers, dropout, learning_rate, weight_decay, out_dir

import os
from torch import Generator

class DataParams():
    def __init__(self, train_csv_path:str, valid_csv_path:str,
                  batch_size:int, num_workers:int, 
                  seed:int, device:str,
                  out_dir:str, maxlen:int):

        assert os.path.exists(train_csv_path), f"Path: {train_csv_path} doesn't exists"
        self.train_csv_path = train_csv_path

        assert os.path.exists(valid_csv_path), f"Path: {valid_csv_path} doesn't exists"
        self.valid_csv_path = valid_csv_path

        assert isinstance(batch_size, int), f"Batch_size must be Integer"
        self.batch_size = batch_size

        assert isinstance(num_workers, int), f"num_workers must be Integer"
        self.num_workers = num_workers

        assert isinstance(seed, int), f"seed must be Integer"
        self.generator = Generator().manual_seed(seed)

        assert device in ['cuda', 'cpu'], f"device should be 'cuda' or 'cpu'"
        self.device = device

        assert isinstance(out_dir, str), f"out_dir must be String"
        self.out_dir = os.path.join(out_dir, 'plots')
        if not os.path.exists(self.out_dir):
            print(f"{self.out_dir} does not exists")
            print(f'Making dirs tree @{self.out_dir}...')
            os.makedirs(self.out_dir, exist_ok=True)
            print('Done.')

        assert isinstance(maxlen, int), f"maxlen must be Integer"
        self.maxlen = maxlen


class TokenizerParams():
    def __init__(self, src_vocab_size:int, trg_vocab_size:int, out_dir:str, lang1_model_prefix:str,
                  lang2_model_prefix:str, lang1_character_coverage:float, lang2_character_coverage:float):

        assert isinstance(src_vocab_size, int), f"src_vocab_size must be Integer"
        self.src_vocab_size = src_vocab_size

        assert isinstance(trg_vocab_size, int), f"trg_vocab_size must be Integer"
        self.trg_vocab_size = trg_vocab_size

        assert isinstance(out_dir, str), f"out_dir must be String"
        self.out_dir = os.path.join(out_dir, 'tokenizers')
        if os.path.exists(self.out_dir):
            print(f"{self.out_dir} does not exists")
            print(f'Making dirs tree @{self.out_dir}...')
            os.makedirs(self.out_dir, exist_ok=True)
            print('Done.')

        assert isinstance(lang1_model_prefix, str), f"lang1_model_prefix must be String"
        self.lang1_model_path = os.path.join(self.out_dir, lang1_model_prefix)

        assert isinstance(lang2_model_prefix, str), f"lang2_model_prefix must be String"
        self.lang2_model_path = os.path.join(self.out_dir, lang2_model_prefix)

        assert isinstance(lang1_character_coverage, float), f"lang1_character_coverage must be Float "
        self.lang1_character_coverage = lang1_character_coverage

        assert isinstance(lang2_character_coverage, float), f"lang2_character_coverage must be Float "
        self.lang2_character_coverage = lang2_character_coverage
        

class ModelParams():
    def __init__(self, model_type:str, epochs:int, dim_embed:int,
                  dim_model:int, dim_feedforward:int, num_layers:int, dropout:float,
                    learning_rate:float, weight_decay:float, out_dir:str):

        assert model_type.lower() in ['s2s', 's2sAttention', 'transformer'], "supported model_type ['s2s', 's2sAttention', 'transformer']."
        self.model_type = model_type.lower()

        assert isinstance(epochs, int), f"epochs must be Integer"
        self.epochs = epochs

        assert isinstance(dim_embed, int), f"dim_embed must be Integer"
        self.dim_embed = dim_embed

        assert isinstance(dim_model, int), f"dim_model must be Integer"
        self.dim_model = dim_model

        assert isinstance(dim_feedforward, int), f"dim_feedforward must be Integer"
        self.dim_feedforward = dim_feedforward

        assert isinstance(num_layers, int), f"num_layers must be Integer"
        self.num_layers = num_layers

        assert isinstance(dropout, float), f"dropout must be Float"
        self.dropout = dropout

        assert isinstance(learning_rate, float), f"learning_rate must be Float"
        self.learning_rate = learning_rate

        assert isinstance(weight_decay, float), f"weight_decay must be Float"
        self.weight_decay = weight_decay

        assert isinstance(out_dir, str), f"out_dir must be String"
        self.out_dir = os.path.join(out_dir, 'models')
        if not os.path.exists(self.out_dir):
            print(f"{self.out_dir} does not exists")
            print(f'Making dirs tree @{self.out_dir}...')
            os.makedirs(self.out_dir, exist_ok=True)
            print('Done.')
        
