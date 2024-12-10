from utils.Data_Parameters import DataParams
from make_model.Model_Parameters import ModelParams
from train import train_model
import os
import argparse
import sys
from config import *
import torch
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
    
    # Call the function with the parsed arguments
    dp = DataParams(train_csv_path=args.train_csv_path,
                    valid_csv_path=args.valid_csv_path,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    device=args.device,
                    out_dir=args.out_dir,
                    maxlen=args.maxlen)
    
    mp = ModelParams(model_type=args.model_type,
                     model_name=args.model_name,
                     out_dir=args.out_dir,
                     dim_embed=args.dim_embed,
                     dim_model=args.dim_model,
                     dim_feedforward=args.dim_feedforward,
                     num_layers=args.num_layers,
                     dropout=args.dropout,
                     learning_rate=args.learning_rate,
                     weight_decay=args.weight_decay)
    
    model = train_model(dp, mp, args.src_tokenizer_path, args.trg_tokenizer_path)
    
    ## Save Entire Model
    model_path = os.path.join(mp.out_dir, mp.model_name)
    torch.save(model, f"{model_path}.bin")
    
    # if True == args.convert_onnx:
    #     print("Converting To ONNX framework...")
    #     onnx_model = utils.model_utils.convert2onxx(model)
    #     onnx_model.save()