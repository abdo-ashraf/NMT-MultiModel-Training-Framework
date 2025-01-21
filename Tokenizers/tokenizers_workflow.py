import argparse
import Tokenizers
import os
import json

#####-----Parameters-----#####
DEFAULT_CONFIG_PATH = 'tokenizers_config.json'
# DEFAULT_LOG_PATH = None


# Command-Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')

    parser.add_argument('--train_csv_path', type=str, required=True, help='CSV of columns "source_lang" and "target_lang" for train')
    parser.add_argument('--config_path', type=str, default=DEFAULT_CONFIG_PATH, help='Path to the config file of the tokenizers')
    # parser.add_argument('--log_path', type=str, default=DEFAULT_LOG_PATH, help='Path to the log file')

    return parser
 

if __name__ == '__main__':
    # Argument parsing and validation
    parser = parse_arguments()
    args = parser.parse_args()
    assert os.path.exists(args.train_csv_path), f"{args.train_csv_path} : Train csv not found."
    assert os.path.exists(args.config_path), f"{args.config_path} : Config file not found."
    
    data = json.load(open(args.config_path, 'r'))
    src_params, tgr_params = data['src_tokenizer'], data['trg_tokenizer']
    
    Tokenizers.train(train_csv_path=args.train_csv_path,
                     src_lang_tokenizer_params=src_params,
                     trg_lang_tokenizer_params=tgr_params)