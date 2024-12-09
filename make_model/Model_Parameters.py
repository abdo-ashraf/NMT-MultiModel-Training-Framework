import os


class ModelParams():
    def __init__(self, model_type:str, dim_embed:int,
                  dim_model:int, dim_feedforward:int, num_layers:int, dropout:float,
                    learning_rate:float, weight_decay:float, out_dir:str):

        assert model_type.lower() in ['s2s', 's2sattention', 'transformer'], "supported model_type ['s2s', 's2sAttention', 'transformer']."
        self.model_type = model_type.lower()

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