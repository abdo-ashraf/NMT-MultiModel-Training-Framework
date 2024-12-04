import argparse
import sys
from data_make import make_data


#####-----Parameters-----#####
DEFAULT_VALID_TEST_SPLIT = 0.4
DEFAULT_SEED = 123
DEFAULT_OUT_DIR = './out/'
DEFAULT_DATA = 'both' ## ('both', 'opus', 'covo')


# Command-Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')

    parser.add_argument('--data', choices=['both', 'opus', 'covo'], default=DEFAULT_DATA, help='type of needed dataset')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR, help='Working dir')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    parser.add_argument('--valid_test_split', type=float, default=DEFAULT_VALID_TEST_SPLIT, help='source character coverage')

    return parser
 

if __name__ == '__main__':
    # Argument parsing and validation
    parser = parse_arguments()
    args = parser.parse_args()

    # Call the function with the parsed arguments
    df_train, df_valid, df_test = make_data(args.out_dir, args.data, args.valid_test_split, args.seed)