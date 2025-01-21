import pandas as pd
from Models.AutoModel import get_model
from Training.Trainer import Trainer
from Training.TrainingArguments import TrainingArguments
from Tokenizers.Tokenizers import Callable_tokenizer
from Models.ModelArgs import ModelArgs
import os
import argparse
import sys
import torch

from utils.data_utils import MT_Dataset, MYCollate
from utils.model_utils import get_parameters_info
# import onnx

## Data params: train_csv_path, valid_csv_path, batch_size, num_workers, seed, device, out_dir, maxlen
## Tokenizer params: src_vocab_size, trg_vocab_size, out_dir, lang1_model_prefix, lang2_model_prefix, lang1_character_coverage, lang2_character_coverage
## Model params: model_type, epochs, dim_embed, dim_model, dim_feedforward, num_layers, dropout, learning_rate, weight_decay, out_dir

# Command-Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Your Program')

    parser.add_argument('--train_csv_path', type=str, required=True, help='CSV of columns "source_lang" and "target_lang" for train')
    parser.add_argument('--valid_csv_path', type=str, required=True, help='CSV of columns "source_lang" and "target_lang" for validation')
    parser.add_argument('--src_tokenizer_path', type=str, required=True, help='Path of source tokenizer model')
    parser.add_argument('--trg_tokenizer_path', type=str, required=True, help='Path of target tokenizer model')
    parser.add_argument('--model_config_path', type=str, required=True, help='A path for model configuration file')
    parser.add_argument('--training_config_path', type=str, required=True, help='A path for training configuration file')
    

    return parser
 

if __name__ == '__main__':
    # Argument parsing and validation
    parser = parse_arguments()
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        parser.error("No argument provided!")

    assert os.path.exists(args.train_csv_path), f"{args.train_csv_path} : Train csv not found."
    assert os.path.exists(args.valid_csv_path), f"{args.valid_csv_path} : Valid csv not found."
    assert os.path.exists(args.src_tokenizer_path), f"{args.src_tokenizer_path} : Tokenizer.model not found."
    assert os.path.exists(args.trg_tokenizer_path), f"{args.trg_tokenizer_path} : Tokenizer.model not found."
    assert os.path.exists(args.model_config_path), f"{args.model_config_path} : Model configuration file not found."
    assert os.path.exists(args.training_config_path), f"{args.training_config_path} : Training configuration file not found."
    

    print("Starting Tokenizers Loading...")
    src_tokenizer = Callable_tokenizer(args.src_tokenizer_path)
    trg_tokenizer = Callable_tokenizer(args.trg_tokenizer_path)
    src_vocab_size = len(src_tokenizer)
    trg_vocab_size = len(trg_tokenizer)
    print("Tokenizers Loading Done.")

    train_df = pd.read_csv(args.train_csv_path)
    valid_df = pd.read_csv(args.valid_csv_path)

    train_ds = MT_Dataset(src_sentences_list=train_df[train_df.columns[0]], trg_sentences_list=train_df[train_df.columns[1]],
                          src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer)
    valid_ds = MT_Dataset(src_sentences_list=valid_df[valid_df.columns[0]], trg_sentences_list=valid_df[valid_df.columns[1]],
                          src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer)
    mycollate = MYCollate(batch_first=True, pad_value=-100)
    
    model_args = ModelArgs(config_path=args.model_config_path)
    model = get_model(model_args, src_vocab_size, trg_vocab_size)

    names, tr, nontr = get_parameters_info(model=model)
    print(f"{'Module':<15}{'Trainable':>15}{'Non-Trainable':>15}")
    for n, ttp, ntp in zip(names, tr, nontr):
        print(f"{n:<15}{ttp:>15,}{ntp:>15,}")
    
    training_args = TrainingArguments(args.training_config_path)

    trainer = Trainer(args=training_args, model=model,
                      train_ds=train_df, valid_ds=valid_df,
                      collator=mycollate,
                      compute_metrics_func=None)
    
    train_losses, valid_losses = trainer.train()
  
