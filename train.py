from make_model.model_make import get_model
from utils.data_utils import load_train_valid, Callable_tokenizer
from utils.model_utils import training, get_parameters_info, plot_loss
from utils.Data_Parameters import DataParams
from make_model.Model_Parameters import ModelParams
import pandas as pd


## Data params: train_csv_path, valid_csv_path, batch_size, num_workers, seed, device, out_dir, maxlen
## Tokenizer params: src_vocab_size, trg_vocab_size, out_dir, lang1_model_prefix, lang2_model_prefix, lang1_character_coverage, lang2_character_coverage
## Model params: model_type, epochs, dim_embed, dim_model, dim_feedforward, num_layers, dropout, learning_rate, weight_decay, out_dir


def train_model(data_params:DataParams, model_params:ModelParams, src_tokenizer_path, trg_tokenizer_path):

  df_train = pd.read_csv(data_params.train_csv_path)
  df_valid = pd.read_csv(data_params.valid_csv_path)

  print("Starting Tokenizers Loading...")
  src_tokenizer = Callable_tokenizer(src_tokenizer_path)
  trg_tokenizer = Callable_tokenizer(trg_tokenizer_path)
  src_pad_tokenId = src_tokenizer.get_tokenId('<pad>')
  print("Tokenizers Loading Done.")

  print("Starting Data Loading...")
  train_loader, valid_loader = load_train_valid(train_sentences=(df_train['source_lang'], df_train['target_lang']),
                                                valid_sentences=(df_valid['source_lang'], df_valid['target_lang']),
                                                src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer,
                                                data_params=data_params)
  print("Data Loading Done.")
  
  print(f"Start Loading {model_params.model_type} Model...")
  model, class_criterion, optim = get_model(params=model_params,
                                            src_vocab_size=len(src_tokenizer),
                                            trg_vocab_size=len(trg_tokenizer),
                                            src_pad_tokenId=src_pad_tokenId,
                                            maxlen=data_params.maxlen)
  print("Model Loading Done.")

  names, tr, nontr = get_parameters_info(model=model)
  print(f"{'Module':<15}{'Trainable':>15}{'Non-Trainable':>15}")
  for n, ttp, ntp in zip(names, tr, nontr):
    print(f"{n:<15}{ttp:>15,}{ntp:>15,}")
  model = model.to(data_params.device)

  print("Starting Model Training...")
  train_class_losses, val_class_losses, model, optim, epochs = training(model=model, criterion=class_criterion,
                                                  optimizer=optim, train_loader=train_loader,
                                                  valid_loader=valid_loader, epochs=data_params.epochs, device=data_params.device)
  print("Model Training Done.")

  plot_loss(train_class_losses, val_class_losses, data_params.plots_dir, model_params.model_name)

  return train_class_losses, val_class_losses, model, optim, epochs
