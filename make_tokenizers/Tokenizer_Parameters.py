import os


class TokenizerParams():
    def __init__(self, src_vocab_size:int, trg_vocab_size:int, out_dir:str, lang1_model_prefix:str,
                  lang2_model_prefix:str, lang1_character_coverage:float, lang2_character_coverage:float):

        assert isinstance(src_vocab_size, int), f"src_vocab_size must be Integer"
        self.src_vocab_size = src_vocab_size

        assert isinstance(trg_vocab_size, int), f"trg_vocab_size must be Integer"
        self.trg_vocab_size = trg_vocab_size

        assert isinstance(out_dir, str), f"out_dir must be String"
        self.out_dir = os.path.join(out_dir, 'tokenizers')
        if not os.path.exists(self.out_dir):
            print(f"{self.out_dir} does not exists")
            print(f'Making dirs tree @{self.out_dir}...')
            os.makedirs(self.out_dir, exist_ok=True)
            print('Done.')

        assert isinstance(lang1_model_prefix, str), f"lang1_model_prefix must be String"
        self.lang1_model_path = os.path.join(self.out_dir, lang1_model_prefix)

        assert isinstance(lang2_model_prefix, str), f"lang2_model_prefix must be String"
        self.lang2_model_path = os.path.join(self.out_dir, lang2_model_prefix)

        assert isinstance(lang1_character_coverage, float), f"lang1_character_coverage must be Float "
        self.lang1_character_coverage = lang1_character_coverage

        assert isinstance(lang2_character_coverage, float), f"lang2_character_coverage must be Float "
        self.lang2_character_coverage = lang2_character_coverage