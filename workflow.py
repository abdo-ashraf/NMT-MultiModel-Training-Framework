import pandas as pd
from Models.AutoModel import get_model
from Training.Trainer import Trainer
from Training.TrainingArguments import TrainingArguments
from Tokenizers.Tokenizers import Callable_tokenizer
from Models.ModelArgs import ModelArgs
import os
import argparse
import sys
from torch.utils.data import DataLoader
from utils import MT_Dataset, MyCollate, compute_metrics, get_parameters_info, plot_history


# Command-Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Your Program')

    parser.add_argument('--train_csv_path', type=str, required=True, help='CSV of columns for train')
    parser.add_argument('--valid_csv_path', type=str, required=True, help='CSV of columns for validation')
    parser.add_argument('--test_csv_path', default='None', type=str, required=False, help='CSV of columns for Testing')
    parser.add_argument('--source_column_name', type=str, required=True, help='source_column_name')
    parser.add_argument('--target_column_name', type=str, required=True, help='target_column_name')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='A path of tokenizer.model')
    parser.add_argument('--model_config_path', type=str, required=True, help='A path for model configuration file')
    parser.add_argument('--training_config_path', type=str, required=True, help='A path for training configuration file')
    parser.add_argument('--out_dir', type=str, required=True, help='A path for output directory')
    parser.add_argument('--model_type', type=str, required=True, choices=['s2s', 's2sAttention', 'transformer'],
                    help='A type of model (choices: s2s, s2sAttention, transformer)')
    
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
    assert os.path.exists(args.tokenizer_path), f"{args.tokenizer_path} : Tokenizer.model not found."
    assert os.path.exists(args.model_config_path), f"{args.model_config_path} : Model configuration file not found."
    assert os.path.exists(args.training_config_path), f"{args.training_config_path} : Training configuration file not found."
    test_csv_path = args.test_csv_path
    if not os.path.exists(test_csv_path):
        print(f"Test_csv path: '{test_csv_path}' does not exists, test_csv_path will set to None")
        test_csv_path = None

    os.makedirs(args.out_dir, exist_ok=True)

    print("---------------------Starting Tokenizer Loading...---------------------")
    tokenizer = Callable_tokenizer(args.tokenizer_path)
    vocab_size = len(tokenizer)
    print(f"Tokenizer length {vocab_size}")
    print("Tokenizer Loading Done.")

    print("---------------------Starting Data Loading...---------------------")
    train_df = pd.read_csv(args.train_csv_path)
    valid_df = pd.read_csv(args.valid_csv_path)

    train_ds = MT_Dataset(input_sentences_list=train_df[args.source_column_name].to_list(),
                            target_sentences_list=train_df[args.target_column_name].to_list(),
                            callable_tokenizer=tokenizer)

    valid_ds = MT_Dataset(input_sentences_list=valid_df[args.source_column_name].to_list(),
                            target_sentences_list=valid_df[args.target_column_name].to_list(),
                            callable_tokenizer=tokenizer)

    mycollate = MyCollate(batch_first=True,
                            pad_value=tokenizer.get_tokenId('<pad>'))

    print(f"Training data length {len(train_ds):,}, Validation data length {len(valid_ds):,}")
    print(f"Source tokens: {train_ds[0][0]}")
    print(f"Target tokens: {train_ds[0][1]}")
    # print(f"Target_loss tokens: {train_ds[0][2]}")
    print("Data Loading Done.")

    print("---------------------Parsing Model arguments...---------------------")
    model_args = ModelArgs(model_type=args.model_type, config_path=args.model_config_path)
    print(model_args)
    print("Parsing Done.")

    print("---------------------Loading the model...---------------------")
    model = get_model(model_args, vocab_size)
    names, tr, nontr = get_parameters_info(model=model)
    print(f"{'Module':<25}{'Trainable':>15}{'Non-Trainable':>15}")
    for n, ttp, ntp in zip(names, tr, nontr):
        print(f"{n:<25}{ttp:>15,}{ntp:>15,}")
    print("Model Loading Done.")
    
    print("---------------------Parsing Training arguments...---------------------")
    training_args = TrainingArguments(args.out_dir, args.training_config_path)
    print(training_args)
    print("Parsing Done.")

    print("---------------------Start training...---------------------")
    trainer = Trainer(args=training_args, model=model,
                        train_ds=train_ds, valid_ds=valid_ds,
                        collator=mycollate, compute_metrics_func=compute_metrics)

    history = trainer.train()
    print(f"Training Done.")

    test_metrics=None
    if test_csv_path is not None:
        print("---------------------Start evaluation on test-set...---------------------")
        test_df = pd.read_csv(test_csv_path)
        test_ds = MT_Dataset(input_sentences_list=test_df[args.source_column_name].to_list(),
                            target_sentences_list=test_df[args.target_column_name].to_list(),
                            callable_tokenizer=tokenizer)

        test_loader = DataLoader(test_ds,
                                batch_size=training_args.batch_size,
                                shuffle=False,
                                collate_fn=mycollate,
                                num_workers=training_args.cpu_num_workers,
                                pin_memory=training_args.pin_memory)
        test_metrics = trainer.evaluate(dataloader=test_loader, set_name='test')
        print(test_metrics)
        print("evaluation Done.")

    save_plots_dir = os.path.join(args.out_dir, 'plots')
    os.makedirs(save_plots_dir, exist_ok=True)
    plot_history(history, test_metrics, save_plots_dir, training_args.run_name)