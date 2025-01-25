import sentencepiece as spm
import pandas as pd
## requirements
# !pip -q install sentencepiece
## source is english and traget is arabic.

def train(train_csv_path:str, tokenizer_params:dict, train_on_columns:list):
  print("Starting Tokenizer Train...")
  
  dataframe = pd.read_csv(train_csv_path)[train_on_columns]
  sentences = []
  for col in train_on_columns:
      sentences = sentences + dataframe[col].to_list()
  spm.SentencePieceTrainer.train(sentence_iterator=iter(sentences), **tokenizer_params)
  
  print(f"{tokenizer_params['model_prefix']} Done")
  statics_dict = {col: len(dataframe[col]) for col in train_on_columns}
  print("Tokenizer Train Done.")
  print(f"Trained on {statics_dict} sentences")


## Tokenizer
class Callable_tokenizer():
    def __init__(self, tokenizer_path):
        self.path = tokenizer_path
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
    def __call__(self, text):
        return self.tokenizer.Encode(text)

    def get_tokenId(self, token_name):
        return self.tokenizer.piece_to_id(token_name)

    def get_tokenName(self, id):
        return self.tokenizer.id_to_piece(id)

    def decode(self, tokens_list):
        return self.tokenizer.Decode(tokens_list)

    def __len__(self):
        return len(self.tokenizer)

    def user_tokenization(self, text):
        return self(text) + [self.get_tokenId('</s>')]
