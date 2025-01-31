seed ?= 123
valid_test_split ?= 0.1
maxlen ?= 25

.PHONY: setup data tokenizer model

setup:
	pip install -r requirements.txt

data:
	@if [ -z "$(data_type)" ]; then \
		echo "Error: data_type is required. Available data types: 1:'ar-en', 2:'sp-en'; \
		exit 1; \
	fi

	@if [ -z "$(out_dir)" ]; then \
		echo "Error: out_dir is required. Please set it."; \
		exit 1; \
	fi

	@if [ "$(data_type)" = "1" ]; then \
	echo "Making ar_en data at $(out_dir)/data/ with seed=$(seed), valid_test_split=$(valid_test_split) and maxlen=$(maxlen)";\
	    python ./make_data/ar_en_data_workflow.py \
	        --out_dir $(out_dir) \
	        --maxlen $(maxlen) \
	        --seed $(seed) \
	        --valid_test_split $(valid_test_split); \
	elif [ "$(data_type)" = "2" ]; then \
	echo "Making sp-en data at $(out_dir)/data/ with seed=$(seed) and valid_test_split=$(valid_test_split)";\
	    python ./make_data/sp_en_data_workflow.py \
	        --out_dir $(out_dir) \
	        --seed $(seed) \
	        --valid_test_split $(valid_test_split); \
	else \
	    echo "Invalid data type. Choose from: 1 (ar-en), 2 (sp-en)"; \
	    exit 1; \
	fi

tokenizer_config_path ?= ./Configurations/tokenizer_config.json
train_csv_path ?= ./data/train.csv
col1 ?= text  # Default column names if not passed
col2 ?= target

tokenizer:
	@echo "Making tokenizer at $(out_dir)/tokenizers/"; \
	python ./Tokenizers/tokenizers_workflow.py \
		--train_csv_path $(train_csv_path) \
		--train_on_columns $(col1) $(col2) \
		--config_path $(tokenizer_config_path) \
		--out_dir $(out_dir)