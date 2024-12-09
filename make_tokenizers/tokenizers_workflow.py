import argparse
from tokenizers_make import train_tokenizers
from Tokenizer_Parameters import TokenizerParams
import os

#####-----Parameters-----#####
DEFAULT_LANG1_MODEL_PREFIX = 'lang1_tokenizer'
DEFAULT_LANG2_MODEL_PREFIX = 'lang2_tokenizer'
DEFAULT_LANG1_CHARACTER_COVERAGE = 0.995
DEFAULT_LANG2_CHARACTER_COVERAGE = 0.995


# Command-Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')

    parser.add_argument('--train_csv_path', type=str, required=True, help='CSV of columns "source_lang" and "target_lang" for train')
    parser.add_argument('--src_vocab_size', type=int, required=True, help='Needed vocabulary size for source')
    parser.add_argument('--trg_vocab_size', type=int, required=True, help='Needed vocabulary size for target')
    parser.add_argument('--out_dir', required=True, help='Working dir')

    parser.add_argument('--lang1_model_prefix', type=str, default=DEFAULT_LANG1_MODEL_PREFIX, help='A prefix for source tokenizer name')
    parser.add_argument('--lang2_model_prefix', type=str, default=DEFAULT_LANG2_MODEL_PREFIX, help='A prefix for target tokenizer name')
    parser.add_argument('--lang1_character_coverage', type=float, default=DEFAULT_LANG1_CHARACTER_COVERAGE, help='source character coverage')
    parser.add_argument('--lang2_character_coverage', type=float, default=DEFAULT_LANG2_CHARACTER_COVERAGE, help='target character coverage')

    return parser
 

if __name__ == '__main__':
    # Argument parsing and validation
    parser = parse_arguments()
    args = parser.parse_args()
    assert os.path.exists(args.train_csv_path), f"{args.train_csv_path} : Train csv not found."

    # Call the function with the parsed arguments
    tp = TokenizerParams(src_vocab_size=args.src_vocab_size,
                         trg_vocab_size=args.trg_vocab_size,
                         out_dir=args.out_dir,
                         lang1_model_prefix=args.lang1_model_prefix,
                         lang2_model_prefix=args.lang2_model_prefix,
                         lang1_character_coverage=args.lang1_character_coverage,
                         lang2_character_coverage=args.lang2_character_coverage)
    
    train_tokenizers(args.train_csv_path, params=tp)