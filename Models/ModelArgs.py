class ModelArgs():
    def __init__(self, model_type:str, model_name:str, dim_embed:int,
                 dim_model:int, dim_feedforward:int, num_layers:int,
                 dropout:float):

        assert model_type.lower() in ['s2s', 's2sattention', 'transformer'], "supported model_type ['s2s', 's2sAttention', 'transformer']."
        self.model_type = model_type.lower()

        assert isinstance(model_name, str), f"model_name must be String"
        self.model_name = model_name
            
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
