from Parameters_Classes import DataParams, TokenizerParams, ModelParams
from train import train
import os
import argparse
import sys
from config import *
# import onnx

## Data params: train_csv_path, valid_csv_path, batch_size, num_workers, seed, device, out_dir, maxlen
## Tokenizer params: src_vocab_size, trg_vocab_size, out_dir, lang1_model_prefix, lang2_model_prefix, lang1_character_coverage, lang2_character_coverage
## Model params: model_type, epochs, dim_embed, dim_model, dim_feedforward, num_layers, dropout, learning_rate, weight_decay, out_dir

# Command-Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Your Program')

    parser.add_argument('--train_csv_path', type=str, required=True, help='CSV of columns "source_lang" and "target_lang" for train')
    parser.add_argument('--valid_csv_path', type=int, required=True, help='CSV of columns "source_lang" and "target_lang" for validation')
    parser.add_argument('--src_vocab_size', type=int, required=True, help='Needed vocabulary size for source')
    parser.add_argument('--trg_vocab_size', type=int, required=True, help='Needed vocabulary size for target')
    parser.add_argument('--out_dir', required=True, help='Working dir')
    parser.add_argument('--model_type', choices=['s2s', 's2sAttention', 'transformer'], required=True, help='needed model architecture')

    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS, help='number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    parser.add_argument('--maxlen', type=int, default=DEFAULT_MAXLEN, help='Maximum length of input sequence')

    parser.add_argument('--lang1_model_prefix', type=str, default=DEFAULT_LANG1_MODEL_PREFIX, help='A prefix for source tokenizer name')
    parser.add_argument('--lang2_model_prefix', type=str, default=DEFAULT_LANG2_MODEL_PREFIX, help='A prefix for target tokenizer name')
    parser.add_argument('--lang1_character_coverage', type=float, default=DEFAULT_LANG1_CHARACTER_COVERAGE, help='source character coverage')
    parser.add_argument('--lang2_character_coverage', type=float, default=DEFAULT_LANG2_CHARACTER_COVERAGE, help='target character coverage')

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

    assert os.access(args.outdir, os.W_OK), f"{args.outdir} : Output Directory has to be writable"
    
    for dir in args.train_csv_path:
        assert os.path.exists(dir), f"{dir} : Train csv not found."

    for dir in args.valid_csv_path:
        assert os.path.exists(dir), f"{dir} : Valid csv not found."
    
    # Call the function with the parsed arguments
    dp = DataParams(args.train_csv_path, args.valid_csv_path, args.batch_size, args.num_workers, args.seed, args.device, args.out_dir, args.maxlen)
    tp = TokenizerParams(args.src_vocab_size, args.trg_vocab_size, args.out_dir, args.lang1_model_prefix, args.lang2_model_prefix, args.lang1_character_coverage, args.lang2_character_coverage)
    mp = ModelParams(args.model_type, args.epochs, args.dim_embed, args.dim_model, args.dim_feedforward, args.num_layers, args.dropout, args.learning_rate, args.weight_decay, args.out_dir)
    model, optim, criterion = train(dp, tp, mp)
    
    # if True == args.convert_onnx:
    #     print("Converting To ONNX framework...")
    #     loaded_model = build_cnn(image_width=args.image_width, image_height=args.image_height,
    #                               seed=args.seed, data_format=args.data_format, compile=True)
    #     loaded_model.load_weights(os.path.join(args.outdir, args.model_name+'.weights.h5'))
    #     onnx_model, _ = tf2onnx.convert.from_keras(loaded_model,opset=14)
    #     onnx.save(onnx_model, os.path.join(args.outdir, args.model_name+".onnx"))