import json

class ModelArgs:
    """
    A class to parse and store model configuration from a JSON file.
    """
    def __init__(self, model_type:str, config_path:str):
        """
        Initialize ModelArgs with configuration from a JSON file.

        Args:
            config_path (str): Path to the JSON configuration file.

        Raises:
            AssertionError: If the JSON content is invalid or has missing keys.
        """
        # Load JSON file
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        # Validate and assign attributes
        self.model_type = model_type.lower()
        assert self.model_type in ['s2s', 's2sattention', 'transformer'], \
            "Supported model_type values are ['s2s', 's2sAttention', 'transformer']."
        
        self.dim_embed = config.get("dim_embed")
        assert isinstance(self.dim_embed, int), "dim_embed must be an integer."
        
        self.dim_model = config.get("dim_model")
        assert isinstance(self.dim_model, int), "dim_model must be an integer."
        
        self.dim_feedforward = config.get("dim_feedforward")
        assert isinstance(self.dim_feedforward, int), "dim_feedforward must be an integer."
        
        self.num_layers = config.get("num_layers")
        assert isinstance(self.num_layers, int), "num_layers must be an integer."
        
        self.dropout = config.get("dropout")
        assert isinstance(self.dropout, float), "dropout must be a float."

        self.maxlen = config.get("maxlen")
        assert isinstance(self.maxlen, int), "maxlen must be an integer."

        self.flash_attention = config.get("flash_attention")
        assert isinstance(self.flash_attention, bool), "flash_attention must be a boolean."

    def __repr__(self):
        return (f"ModelArgs(\n" +
                f"model_type={self.model_type},\n" +
                f"dim_embed={self.dim_embed},\n" +
                f"dim_model={self.dim_model},\n" +
                f"dim_feedforward={self.dim_feedforward},\n" +
                f"num_layers={self.num_layers},\n" +
                f"dropout={self.dropout},\n" +
                f"maxlen={self.maxlen},\n" +
                f"flash_attention={self.flash_attention}\n" +
                ")")