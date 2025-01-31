import argparse
import Tokenizers
import os
import json

#####-----Parameters-----#####
DEFAULT_CONFIG_PATH = './Configurations/tokenizers_config.json'
# DEFAULT_LOG_PATH = None


# Command-Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')

    parser.add_argument('--out_dir', type=str, required=True, help='A path for output directory')
    parser.add_argument('--train_csv_path', type=str, required=True, help='CSV of columns "source_lang" and "target_lang" for train')
    parser.add_argument('--train_on_columns', type=str, nargs='+', required=True, help='List of columns to train the tokenizer on')
    parser.add_argument('--config_path', type=str, default=DEFAULT_CONFIG_PATH, help='Path to the config file of the tokenizers')
    # parser.add_argument('--log_path', type=str, default=DEFAULT_LOG_PATH, help='Path to the log file')
    return parser


if __name__ == '__main__':
    # Argument parsing and validation
    parser = parse_arguments()
    args = parser.parse_args()
    assert os.path.exists(args.train_csv_path), f"{args.train_csv_path} : Train csv not found."
    assert os.path.exists(args.config_path), f"{args.config_path} : Config file not found."
    tokenizer_path = os.path.join(args.out_dir, "tokenizers")
    os.makedirs(tokenizer_path, exist_ok=True)
    
    tokenizer_params = json.load(open(args.config_path, 'r'))
    Tokenizers.train(train_csv_path=args.train_csv_path,
                     tokenizer_params=tokenizer_params,
                     train_on_columns=args.train_on_columns,
                     tokenizer_path=tokenizer_path)