from models.seq2seq_model import Seq2seq_no_attention
from models.seq2seqAttention_model import Seq2seq_with_attention
from models.Transformer_model import NMT_Transformer
from utils.data_utils import load_train_valid, tokenizers_train, Callable_tokenizer
from utils.model_utils import loss_acc_loader, training, get_parameters_info, plot_loss
from utils.Parameters_Classes import DataParams, TokenizerParams, ModelParams 
import pandas as pd
import torch
from torch import nn

## Data params: train_csv_path, valid_csv_path, batch_size, num_workers, seed, device, out_dir, maxlen
## Tokenizer params: src_vocab_size, trg_vocab_size, out_dir, lang1_model_prefix, lang2_model_prefix, lang1_character_coverage, lang2_character_coverage
## Model params: model_type, epochs, dim_embed, dim_model, dim_feedforward, num_layers, dropout, learning_rate, weight_decay, out_dir

def train(data_params:DataParams, tokenizer_params:TokenizerParams, model_params:ModelParams):

  df_train = pd.read_csv(data_params.train_csv_path)
  df_valid = pd.read_csv(data_params.valid_csv_path)

  print("Starting Tokenizers Train...")
  tokenizers_train(df_train['source_lang'], df_train['target_lang'], tokenizer_params)
  print("Tokenizers Train Done.")

  print("Starting Data Loading...")
  train_loader, valid_loader = load_train_valid(train_sentences=(df_train['source_lang'], df_train['target_lang']),
                                                valid_sentences=(df_valid['source_lang'], df_valid['target_lang']),
                                                tokenizer_params=tokenizer_params,
                                                data_params=data_params)
  print("Data Loading Done.")

  src_tokenizer = Callable_tokenizer(tokenizer_params.lang1_model_path)
  src_pad_tokenId = src_tokenizer.get_tokenId('<pad>')
  
  print("Starting Model Loading...")
  if model_params.model_type == 's2s': model = Seq2seq_no_attention(encoder_vocab_size=tokenizer_params.src_vocab_size,
                                                        decoder_vocab_size=tokenizer_params.trg_vocab_size,
                                                        dim_embed=model_params.dim_embed,
                                                        dim_model=model_params.dim_model,
                                                        num_layers=model_params.num_layers,
                                                        dropout_probability=model_params.dropout)
      
  elif model_params.model_type == 's2sAttention': model = Seq2seq_with_attention(encoder_vocab_size=tokenizer_params.src_vocab_size,
                                                                                 decoder_vocab_size=tokenizer_params.trg_vocab_size,
                                                                                 dim_embed=model_params.dim_embed,
                                                                                 dim_model=model_params.dim_model,
                                                                                 num_layers=model_params.num_layers,
                                                                                 dropout_probability=model_params.dropout)

  else: model = NMT_Transformer(encoder_vocab_size=tokenizer_params.src_vocab_size,
                                decoder_vocab_size=tokenizer_params.trg_vocab_size,
                                dim_embed=model_params.dim_embed,
                                dim_model=model_params.dim_model,
                                num_layers=model_params.num_layers,
                                dropout_probability=model_params.dropout,
                                src_pad_tokenId=src_pad_tokenId,
                                maxlen=data_params.maxlen)
  print("Model Loading Done.")

  tr, nontr = get_parameters_info(model=model)
  print(f"Total trainable parameters = {tr:,}\nTotal non-trainable parameters = {nontr:,}")
  model = model.to(data_params.device)

  class_criterion = nn.CrossEntropyLoss(ignore_index=src_pad_tokenId)  # Classification loss
  optim = torch.optim.AdamW(params=model.parameters(),
                            lr=model_params.learning_rate,
                            weight_decay=model_params.weight_decay)
  
  print("Starting Model Training...")
  train_class_losses, val_class_losses = training(model=model, criterion=class_criterion,
                                                  optimizer=optim, train_loader=train_loader,
                                                  valid_loader=valid_loader, epochs=model_params.epochs, device=data_params.device)
  print("Model Training Done.")

  plot_loss(train_class_losses, val_class_losses, data_params.plots_dir)

  return model, optim, class_criterion

def testing(test_csv_path, model):
  pass
