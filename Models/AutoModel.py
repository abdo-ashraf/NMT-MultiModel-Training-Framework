from Models.seq2seq_model import Seq2seq_no_attention
from Models.seq2seqAttention_model import Seq2seq_with_attention
from Models.Transformer_model import NMT_Transformer
from Models.ModelArgs import ModelArgs


def get_model(params:ModelArgs, vocab_size):

    if params.model_type.lower() == 's2s': model = Seq2seq_no_attention(vocab_size=vocab_size,
                                                        dim_embed=params.dim_embed,
                                                        dim_model=params.dim_model,
                                                        dim_feedforward=params.dim_feedforward,
                                                        num_layers=params.num_layers,
                                                        dropout_probability=params.dropout)
      
    elif params.model_type.lower() == 's2sattention': model = Seq2seq_with_attention(vocab_size=vocab_size,
                                                                                 dim_embed=params.dim_embed,
                                                                                 dim_model=params.dim_model,
                                                                                 dim_feedforward=params.dim_feedforward,
                                                                                 num_layers=params.num_layers,
                                                                                 dropout_probability=params.dropout)

    else: model = NMT_Transformer(vocab_size=vocab_size,
                                dim_embed=params.dim_embed,
                                dim_model=params.dim_model,
                                dim_feedforward=params.dim_feedforward,
                                num_layers=params.num_layers,
                                dropout_probability=params.dropout,
                                maxlen=params.maxlen)
    return model
    