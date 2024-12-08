import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from torch import nn
from utils.Parameters_Classes import DataParams, TokenizerParams 
## requirements
# !pip -q install sentencepiece


def tokenizers_train(lang1_sentences, lang2_sentences, params:TokenizerParams):

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(lang1_sentences),
        model_prefix=params.lang1_model_path, ## Prefix for saved model files
        vocab_size=params.src_vocab_size,
        character_coverage=params.lang1_character_coverage,
        model_type="bpe",        ## Using Byte Pair Encoding (BPE) model for subword tokenization
        pad_id=0,                ## ID for <pad> (pad token)
        unk_id=1,                ## ID for <unk> (unknown token)
        bos_id=2,                ## ID for <s> (beginning of sentence token)
        eos_id=3                 ## ID for </s> (end of sentence token)
    )
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
        eos_id=3
    )
    print(f'{params.lang2_model_path} Done')



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

## Dataset
class MT_Dataset(Dataset):

    def __init__(self, src_sentences_list, trg_sentences_list, src_tokenizer:Callable_tokenizer, trg_tokenizer:Callable_tokenizer, reversed_input=False):
        super(MT_Dataset, self).__init__()

        assert len(src_sentences_list) == len(trg_sentences_list), (f"Lengths mismatched: input has {len(src_sentences_list)} sentences, "f"but target has {len(trg_sentences_list)} sentences.")
        
        self.src_sentences_list = src_sentences_list
        self.trg_sentences_list = trg_sentences_list
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.reversed_input = reversed_input
        # self.maxlen = maxlen

    def __len__(self):
        return len(self.src_sentences_list)

    def __getitem__(self, index):
        input, target = self.src_sentences_list[index], self.trg_sentences_list[index]

        input_numrical_tokens = [self.src_tokenizer.get_tokenId('<s>')] + self.src_tokenizer(input) + [self.src_tokenizer.get_tokenId('</s>')]
        target_numrical_tokens = [self.trg_tokenizer.get_tokenId('<s>')] +  self.trg_tokenizer(target) + [self.trg_tokenizer.get_tokenId('</s>')]
        
        input_tensor_tokens = torch.tensor(input_numrical_tokens)
        target_tensor_tokens = torch.tensor(target_numrical_tokens)

        if self.reversed_input: input_tensor_tokens = input_tensor_tokens.flip(0)

        return input_tensor_tokens, target_tensor_tokens
    
## Collate
class MYCollate():
    def __init__(self, batch_first=True, pad_value=0):
        self.pad_value = pad_value
        self.batch_first = batch_first

    def __call__(self, data):
        en_stentences = [ex[0] for ex in data]
        ar_stentences = [ex[1] for ex in data]

        padded_en_stentences = nn.utils.rnn.pad_sequence(en_stentences, batch_first=self.batch_first,
                                                      padding_value=self.pad_value)
        padded_ar_stentences = nn.utils.rnn.pad_sequence(ar_stentences, batch_first=self.batch_first,
                                                      padding_value=self.pad_value)
        return padded_en_stentences, padded_ar_stentences


def load_train_valid(train_sentences, valid_sentences, tokenizer_params:TokenizerParams, data_params:DataParams):
    src_tokenizer = Callable_tokenizer(tokenizer_params.lang1_model_path)
    trg_tokenizer = Callable_tokenizer(tokenizer_params.lang2_model_path)

    train_ds = MT_Dataset(src_sentences_list=train_sentences[0], trg_sentences_list=train_sentences[1],
                          src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer)
    
    valid_ds = MT_Dataset(src_sentences_list=valid_sentences[0], trg_sentences_list=valid_sentences[1],
                          src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer)
    
    mycollate = MYCollate(batch_first=True, pad_value=trg_tokenizer.get_tokenId('<pad>'))
    pin_memory = True if data_params.device == 'cuda' else False

    train_loader = DataLoader(train_ds, batch_size=data_params.batch_size, shuffle=True,
                          collate_fn=mycollate, num_workers=data_params.num_workers,
                          generator=data_params.generator, pin_memory=pin_memory)

    valid_loader = DataLoader(valid_ds, batch_size=data_params.batch_size, shuffle=False,
                            collate_fn=mycollate, num_workers=data_params.num_workers,
                            generator=data_params.generator, pin_memory=pin_memory)
    
    return train_loader, valid_loader