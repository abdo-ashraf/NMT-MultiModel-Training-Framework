import torch
from torch import nn


class NMT_Transformer(nn.Module):
    def __init__(self, encoder_vocab_size:int, decoder_vocab_size:int,
                 dim_embed:int, dim_model:int, dim_feedforward:int,
                 num_layers:int, dropout_probability:float, maxlen:int):
        super().__init__()

        self.src_embed = nn.Embedding(num_embeddings=encoder_vocab_size, embedding_dim=dim_embed)
        self.src_pos = nn.Embedding(num_embeddings=maxlen, embedding_dim=dim_embed)

        self.trg_embed = nn.Embedding(num_embeddings=decoder_vocab_size, embedding_dim=dim_embed)
        self.trg_pos = nn.Embedding(num_embeddings=maxlen, embedding_dim=dim_embed)

        self.dropout = nn.Dropout(dropout_probability)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=8,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout_probability,
                                                   batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=8,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout_probability,
                                                   batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        self.classifier = nn.Linear(dim_model, decoder_vocab_size)
        self.maxlen = maxlen
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, source, target, src_pad_tokenId=None, trg_pad_tokenId=None):
        B, Ts = source.shape
        B, Tt = target.shape
        device = source.device
        ## Encoder Path
        src_poses = self.src_pos(torch.arange(0, Ts).to(device).unsqueeze(0).repeat(B, 1))
        src_embedings = self.dropout(self.src_embed(source) + src_poses)

        src_pad_mask = source == src_pad_tokenId if src_pad_tokenId is not None else None
        memory = self.transformer_encoder(src=src_embedings, mask=None, src_key_padding_mask=src_pad_mask, is_causal=False)
        ## Decoder Path
        trg_poses = self.trg_pos(torch.arange(0, Tt).to(device).unsqueeze(0).repeat(B, 1))
        trg_embedings = self.dropout(self.trg_embed(target) + trg_poses)
        
        trg_pad_mask = target == trg_pad_tokenId if trg_pad_tokenId is not None else None
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(Tt, dtype=bool).to(device)
        decoder_out = self.transformer_decoder.forward(tgt=trg_embedings,
                                                memory=memory,
                                                tgt_mask=tgt_mask,
                                                memory_mask=None,
                                                tgt_key_padding_mask=trg_pad_mask,
                                                memory_key_padding_mask=None)
        ## Classifier Path
        logits = self.classifier(decoder_out)

        # return logits, decoder_out, tgt_mask, trg_pad_mask, memory, src_pad_mask, src_poses, src_embedings, trg_poses, trg_embedings
        return logits
    
    @torch.no_grad
    def translate(self, source: torch.Tensor, trg_sos_tokenId: int, trg_eos_tokenId:int, max_tries=100):
        """
        Translates a source sequence into a target sequence using greedy decoding.
        """
        
        source = source.unsqueeze(0)
        B, Ts = source.shape
        device = source.device
        trg_tokens = torch.tensor([trg_sos_tokenId]).unsqueeze(0).to(device)

        for _ in range(max_tries):
            logits = self.forward(source=source, target=trg_tokens)[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)  # Greedy decoding
            # Append predicted token
            trg_tokens = torch.cat([trg_tokens, next_token], dim=1)

            # Stop if all batches predict <EOS> (commonly token ID 3)
            if torch.all(next_token == trg_eos_tokenId):
                break
        return trg_tokens.squeeze(0).tolist()