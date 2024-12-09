import torch
from make_model.seq2seq_model import Seq2seq_no_attention
from make_model.seq2seqAttention_model import Seq2seq_with_attention
from make_model.Transformer_model import NMT_Transformer
from make_model.Model_Parameters import ModelParams


def get_model(params:ModelParams, src_vocab_size, trg_vocab_size, src_pad_tokenId, maxlen):

    if params.model_type.lower() == 's2s': model = Seq2seq_no_attention(encoder_vocab_size=src_vocab_size,
                                                        decoder_vocab_size=trg_vocab_size,
                                                        dim_embed=params.dim_embed,
                                                        dim_model=params.dim_model,
                                                        dim_feedforward=params.dim_feedforward,
                                                        num_layers=params.num_layers,
                                                        dropout_probability=params.dropout)
      
    elif params.model_type.lower() == 's2sattention': model = Seq2seq_with_attention(encoder_vocab_size=src_vocab_size,
                                                                                 decoder_vocab_size=trg_vocab_size,
                                                                                 dim_embed=params.dim_embed,
                                                                                 dim_model=params.dim_model,
                                                                                 dim_feedforward=params.dim_feedforward,
                                                                                 num_layers=params.num_layers,
                                                                                 dropout_probability=params.dropout)

    else: model = NMT_Transformer(encoder_vocab_size=src_vocab_size,
                                decoder_vocab_size=trg_vocab_size,
                                dim_embed=params.dim_embed,
                                dim_model=params.dim_model,
                                dim_feedforward=params.dim_feedforward,
                                num_layers=params.num_layers,
                                dropout_probability=params.dropout,
                                src_pad_tokenId=src_pad_tokenId,
                                maxlen=maxlen)
    
    class_criterion = torch.nn.CrossEntropyLoss(ignore_index=src_pad_tokenId)  # Classification loss
    optim = torch.optim.AdamW(params=model.parameters(),
                            lr=params.learning_rate,
                            weight_decay=params.weight_decay)
    
    return model, class_criterion, optim