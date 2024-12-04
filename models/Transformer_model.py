import torch
from torch import nn


class NMT_Transformer(nn.Module):
    def __init__(self, encoder_vocab_size:int, decoder_vocab_size:int,
                 dim_embed:int, dim_model:int, dim_feedforward:int,
                 num_layers:int, dropout_probability:float,
                 src_pad_tokenId:int, maxlen:int):
        super().__init__()

        self.src_embed = nn.Embedding(num_embeddings=encoder_vocab_size, embedding_dim=dim_embed)
        self.src_pos = nn.Embedding(num_embeddings=maxlen, embedding_dim=dim_embed)

        self.trg_embed = nn.Embedding(num_embeddings=decoder_vocab_size, embedding_dim=dim_embed)
        self.trg_pos = nn.Embedding(num_embeddings=maxlen, embedding_dim=dim_embed)

        self.dropout = nn.Dropout(dropout_probability)

        self.tranformer = nn.Transformer(d_model=dim_model, nhead=8,
                                         num_encoder_layers=num_layers,
                                         num_decoder_layers=num_layers,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout_probability,
                                         batch_first=True,
                                         norm_first=False)
        
        self.classifier = nn.Linear(dim_model, decoder_vocab_size)
        self.make_src_mask = lambda src: src==src_pad_tokenId

    def forward(self, source, target):
        B, Ts = source.shape
        B, Tt = target.shape
        device = source.device

        src_poses = self.src_pos(torch.arange(0, Ts).to(device).unsqueeze(0).repeat(B, 1))
        src_embedings = self.dropout(self.src_embed(source) + src_poses)

        trg_poses = self.trg_pos(torch.arange(0, Tt).to(device).unsqueeze(0).repeat(B, 1))
        trg_embedings = self.dropout(self.trg_embed(source) + trg_poses)

        src_mask = self.make_src_mask(source).to(device)
        trg_mask = self.tranformer.generate_square_subsequent_mask(Tt).to(device)

        tf_out = self.tranformer(src_embedings, trg_embedings, src_key_padding_mask=src_mask, tgt_mask=trg_mask)
        out = self.classifier(tf_out)

        return out

    def translate(self, source:torch.Tensor, sos_tokenId, max_tries=50):
        pass

