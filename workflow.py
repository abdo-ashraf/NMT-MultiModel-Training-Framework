import pandas as pd
from Models.AutoModel import get_model
from Models.Trainer import Trainer
from Models.TrainingArguments import TrainingArguments
from Tokenizers.Tokenizers import Callable_tokenizer
from Models.ModelArgs import ModelArgs
from train import train_model
import os
import argparse
import sys
from config import *
import torch

from utils.data_utils import MT_Dataset, MYCollate
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
    parser.add_argument('--out_dir', type=str, required=True, help='Working dir')
    parser.add_argument('--model_type', choices=['s2s', 's2sAttention', 'transformer'], required=True, help='needed model architecture')
    parser.add_argument('--model_name', type=str, required=True, help='Model prefix name')

    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS, help='number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    parser.add_argument('--maxlen', type=int, default=DEFAULT_MAXLEN, help='Maximum length of input sequence')

    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='number of training epochs')
    parser.add_argument('--dim_embed', type=int, default=DEFAULT_DIM_EMBED, help='dims of embedding matrix')
    parser.add_argument('--dim_model', type=int, default=DEFAULT_DIM_MODEL, help='dims of the model')
    parser.add_argument('--dim_feedforward', type=int, default=DEFAULT_DIM_FEEDFORWARD, help='dims of feedforward network at the model')
    parser.add_argument('--num_layers', type=int, default=DEFAULT_NUM_LAYERS, help='number of layers for encoder and decoder')
    parser.add_argument('--dropout', type=float, default=DEFAULT_DROPOUT, help='proba of all dropout layers')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY, help='weight decay')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=DEFAULT_DEVICE, help='needed model architecture')
    parser.add_argument('--in_onnx', type=bool, default=DEFAULT_IN_ONNX, help='Save the model in onnx (.onnx) or in pytorch (.pth) format')

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
    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir, exist_ok=True)
    


    print("Starting Tokenizers Loading...")
    src_tokenizer = Callable_tokenizer(args.src_tokenizer_path)
    trg_tokenizer = Callable_tokenizer(args.trg_tokenizer_path)
    src_vocab_size = len(src_tokenizer)
    trg_vocab_size = len(trg_tokenizer)
    print("Tokenizers Loading Done.")

    train_df = pd.read_csv(args.train_csv_path)
    valid_df = pd.read_csv(args.valid_csv_path)

    train_ds = MT_Dataset(src_sentences_list=train_df[0], trg_sentences_list=train_df[1],
                          src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer)
    valid_ds = MT_Dataset(src_sentences_list=valid_df[0], trg_sentences_list=valid_df[1],
                          src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer)
    
    mycollate = MYCollate(batch_first=True, pad_value=-100)
    
    model_args = ModelArgs(model_type=args.model_type,
                           model_name=args.model_name,
                           dim_embed=args.dim_embed,
                           dim_model=args.dim_model,
                           dim_feedforward=args.dim_feedforward,
                           num_layers=args.num_layers,
                           dropout=args.dropout)
    model = get_model(model_args, src_vocab_size, trg_vocab_size, -100, args.maxlen)

    training_args = TrainingArguments()
    trainer = Trainer(args=training_args, model=model,
                      train_ds=train_df, valid_ds=valid_df,
                      collator=MYCollate(), compute_metrics_func=None)
    
    train_losses, valid_losses = trainer.train()



    # train_class_losses, val_class_losses, model, optim, epochs = train_model(dp, args.src_tokenizer_path, args.trg_tokenizer_path)
    
    
        
