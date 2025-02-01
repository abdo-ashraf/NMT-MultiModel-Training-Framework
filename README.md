# NMT-MultiModel-Training-Framework
**NMT-MultiModel-Training-Framework** is a versatile and scalable framework designed for training and evaluating Neural Machine Translation (NMT) models using multiple architectures. This framework supports various NMT models, including but not limited to Seq2Seq, Transformer, and Attention-based models. It is built to facilitate easy experimentation, customization, and deployment of NMT systems.

Whether you're a researcher exploring new NMT architectures or a developer building production-ready translation systems, this framework provides the tools and flexibility to meet your needs.

---

## Features

- **Multi-Model Support**: Train and evaluate multiple NMT architectures (e.g., Seq2Seq, Transformer) within a single framework.
- **Customizable Configurations**: Easily configure model hyperparameters, data preprocessing, and training pipelines.
- **Data Preprocessing Tools**: Built-in tools for tokenization, batching, and dataset preparation.
- **Evaluation Metrics**: Compute standard NMT evaluation metrics such as BLEU.
- **Scalable Training**: Supports training on both CPU and GPU, with options for distributed training.
- **Extensible Design**: Modular codebase for adding new models, datasets, or evaluation metrics.

---

## Installation

### Prerequisites

- Python 3.11 or higher
- PyTorch (>=  2.4.1)
- Other dependencies listed in `requirements.txt`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/abdo-ashraf/NMT-MultiModel-Training-Framework.git
   cd NMT-MultiModel-Training-Framework
   ```
2. Install the required dependencies:
   ```bash
   make setup
   ```
   
---

## Usage
1. Prepare Your Dataset:
   You can use your custom dataset or generate data using the following command:
   ```bash
   make data data_type=1 out_dir=./out
   ```
   Currently, the available `data_type` options are:
   - `"en-ar": 1` (English to Arabic)
   - `"en-sp": 2` (English to Spanish)

   The generated data will be stored in `/out_dir/data/`.
   
3. Prepare Tokenizer Configurations:
   
   Modify the configuration file at: `/Configurations/tokenizer_config.json` to specify tokenizer settings such as vocabulary size, special tokens, and tokenization method.

5. Train a Tokenizer:
   Train a SentencePiece tokenizer on the training dataset using the following command:
   ```bash
   make tokenizer \
      train_csv_path=/out/data/en-ar_train.csv \
      train_col1=ar train_col2=en \
      tokenizer_config_path=/Configurations/tokenizer_config.json \
      out_dir=./out
   ```
   This command trains a tokenizer using the specified training dataset (`train_csv_path`) and columns (`train_col1` and `train_col2`). The tokenizer's configuration (e.g., vocabulary size, special tokens) is defined in `tokenizer_config.json`, and the trained tokenizer is saved in the `out_dir/tokenizers/` directory.

6. Prepare Model Configurations:
   
   Before training your NMT model, you can customize its architecture and hyperparameters by modifying the configuration file located at: `/Configurations/model_config.json` TThis file contains default values for all parameters, which are pre-configured to work well for most use cases. However, you can adjust these settings based on your specific requirements or leave them as their default values if you prefer. The available parameters include:
   - `dim_embed`: The dimensionality of the token embeddings, which maps input tokens to dense vectors.
   - `dim_model`: The dimensionality of the model's hidden states, determining the size of the encoder and decoder layers.
   - `dim_feedforward`: The dimensionality of the feedforward network's inner layer within the Transformer architecture.
   - `num_layers`: The number of layers in both the encoder and decoder stacks.
   - `dropout`: The dropout rate to prevent overfitting during training.
   - `maxlen`: The maximum sequence length for input and output tokens, ensuring consistent tensor shapes.
   - `flash_attention`: A boolean flag to enable or disable Flash Attention, an optimized attention mechanism for faster training on supported hardware.
     
   **Note**: Flash Attention is not yet available for use and will be added in a future update.

Adjust these parameters based on your dataset size, computational resources, and desired model performance. Once configured, the framework will use these settings to initialize and train your NMT model.

## Deployment Link for: https://huggingface.co/spaces/TheDemond/Neural-machine-translation
