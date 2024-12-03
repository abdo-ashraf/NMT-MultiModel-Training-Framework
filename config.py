## Data params: train_csv_path, valid_csv_path, batch_size, num_workers, seed, device, out_dir, maxlen
## Tokenizer params: src_vocab_size, trg_vocab_size, out_dir, lang1_model_prefix, lang2_model_prefix, lang1_character_coverage, lang2_character_coverage
## Model params: model_type, epochs, dim_embed, dim_model, dim_feedforward, num_layers, dropout, learning_rate, weight_decay, out_dir

## All needed params
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 20
DEFAULT_NUM_WORKERS = 2
DEFAULT_SEED = 123
DEFAULT_MAXLEN = 100
DEFAULT_LANG1_MODEL_PREFIX = 'lang1_tokenizer'
DEFAULT_LANG2_MODEL_PREFIX = 'lang2_tokenizer'
DEFAULT_LANG1_CHARACTER_COVERAGE = 0.995
DEFAULT_LANG2_CHARACTER_COVERAGE = 0.995
DEFAULT_DIM_EMBED = 512
DEFAULT_DIM_MODEL = 512
DEFAULT_DIM_FEEDFORWARD = DEFAULT_DIM_MODEL*4
DEFAULT_NUM_LAYERS = 6
DEFAULT_DROPOUT = 0.3
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_DEVICE = 'cpu'

# # Default ONNX model opset version.
# DEFAULT_CONVERT_TO_ONNX = "False"
# DEFAULT_ONNX_MODEL_OPSET_VERSION = 14