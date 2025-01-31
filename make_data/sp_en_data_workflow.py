import argparse
from sp_en_data_make import sp_en_data
import os

#####-----Parameters-----#####
# DEFAULT_OUT_DIR = './out/'
DEFAULT_SEED = 123
DEFAULT_VALID_TEST_SPLIT = 0.3

# Command-Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')

    parser.add_argument('--out_dir', required=True, help='Working dir')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    parser.add_argument('--valid_test_split', type=float, default=DEFAULT_VALID_TEST_SPLIT, help='source character coverage')

    return parser
 

if __name__ == '__main__':
    # Argument parsing and validation
    parser = parse_arguments()
    args = parser.parse_args()

    data_dir = os.path.join(args.out_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Call the function with the parsed arguments
    df_train, df_valid, df_test = sp_en_data(data_dir, args.valid_test_split, args.seed)