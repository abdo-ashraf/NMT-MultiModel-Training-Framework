import sentencepiece as spm
import pandas as pd
## requirements
# !pip -q install sentencepiece
## source is english and traget is arabic.

def train(train_csv_path:str, src_lang_tokenizer_params:dict, trg_lang_tokenizer_params:dict):
  print("Starting Tokenizers Train...")
  
  df_train = pd.read_csv(train_csv_path)
  src_sentences, trg_sentences = df_train['source_lang'], df_train['target_lang']

  spm.SentencePieceTrainer.train(
      sentence_iterator=iter(src_sentences), **src_lang_tokenizer_params)
  print(f"{src_lang_tokenizer_params['model_prefix']} Done")

  spm.SentencePieceTrainer.train(
      sentence_iterator=iter(trg_sentences), **trg_lang_tokenizer_params)
  print(f"{trg_lang_tokenizer_params['model_prefix']} Done")

  print("Tokenizers Train Done.")


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
        return [self.get_tokenId('<s>')] + self(text) + [self.get_tokenId('</s>')]
