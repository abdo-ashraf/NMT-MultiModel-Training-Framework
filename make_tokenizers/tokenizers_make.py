import sentencepiece as spm
from Tokenizer_Parameters import TokenizerParams
import pandas as pd
## requirements
# !pip -q install sentencepiece


def train_tokenizers(train_csv_path:str, params:TokenizerParams):
  print("Starting Tokenizers Train...")
  
  df_train = pd.read_csv(train_csv_path)
  lang1_sentences, lang2_sentences = df_train['source_lang'], df_train['target_lang']

  spm.SentencePieceTrainer.train(
      sentence_iterator=iter(lang1_sentences),
      model_prefix=params.lang1_model_path, ## Prefix for saved model files
      vocab_size=params.src_vocab_size,
      character_coverage=params.lang1_character_coverage,
      model_type="bpe",        ## Using Byte Pair Encoding (BPE) model for subword tokenization
      pad_id=0,                ## ID for <pad> (pad token)
      unk_id=1,                ## ID for <unk> (unknown token)
      bos_id=2,                ## ID for <s> (beginning of sentence token)
      eos_id=3)                ## ID for </s> (end of sentence token)
  print(f'{params.lang1_model_path} Done')

  spm.SentencePieceTrainer.train(
      sentence_iterator=iter(lang2_sentences), 
      model_prefix=params.lang2_model_path,
      vocab_size=params.trg_vocab_size,
      character_coverage=params.lang2_character_coverage,
      model_type="bpe",
      pad_id=0,
      unk_id=1,
      bos_id=2,
      eos_id=3)
  print(f'{params.lang2_model_path} Done')

  print("Tokenizers Train Done.")