# Set default for optional test_csv_path if not provided
seed ?= 123
valid_test_split ?= 0.1
maxlen ?= 25
test_csv_path ?= None

.PHONY: setup data tokenizer model

setup:
	pip install -r requirements.txt

data:
# Check for required parameters
	@if [ -z "$(data_type)" ]; then \
		echo "Error: data_type is required. Available data types: 1:'ar-en', 2:'sp-en'"; \
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


tokenizer:
# Check for required parameters
	@if [ -z "$(train_csv_path)" ]; then \
		echo "Error: train_csv_path is required."; \
		exit 1; \
	fi
	@if [ -z "$(train_col1)" ]; then \
		echo "Error: train_col1 is required."; \
		exit 1; \
	fi
	@if [ -z "$(train_col2)" ]; then \
		echo "Error: train_col2 is required."; \
		exit 1; \
	fi
	@if [ -z "$(tokenizer_config_path)" ]; then \
		echo "Error: tokenizer_config_path is required."; \
		exit 1; \
	fi
	@if [ -z "$(out_dir)" ]; then \
		echo "Error: out_dir is required."; \
		exit 1; \
	fi

	@echo "Making tokenizer at $(out_dir)/tokenizers/ ";
	@echo "with configurations at $(tokenizer_config_path) ";
	@echo "on Columns '$(train_col1)', '$(train_col2)' ";
	@echo "of $(train_csv_path) csv.";
	@python ./Tokenizers/tokenizers_workflow.py \
		--train_csv_path $(train_csv_path) \
		--train_on_columns $(train_col1) $(train_col2) \
		--config_path $(tokenizer_config_path) \
		--out_dir $(out_dir)

model:
# Check for required parameters
	@if [ -z "$(train_csv_path)" ]; then \
		echo "Error: train_csv_path is required."; \
		exit 1; \
	fi
	@if [ -z "$(valid_csv_path)" ]; then \
		echo "Error: valid_csv_path is required."; \
		exit 1; \
	fi
	@if [ -z "$(source_column_name)" ]; then \
		echo "Error: source_column_name is required."; \
		exit 1; \
	fi
	@if [ -z "$(target_column_name)" ]; then \
		echo "Error: target_column_name is required."; \
		exit 1; \
	fi
	@if [ -z "$(tokenizer_path)" ]; then \
		echo "Error: tokenizer_path is required."; \
		exit 1; \
	fi
	@if [ -z "$(model_config_path)" ]; then \
		echo "Error: model_config_path is required."; \
		exit 1; \
	fi
	@if [ -z "$(training_config_path)" ]; then \
		echo "Error: training_config_path is required."; \
		exit 1; \
	fi
	@if [ -z "$(out_dir)" ]; then \
		echo "Error: out_dir is required."; \
		exit 1; \
	fi
	@if [ -z "$(model_type)" ]; then \
		echo "Error: model_type is required."; \
		exit 1; \
	fi

	@echo "Making $(model_type) model at $(out_dir)/models/ with ";
	@echo "train_csv_path=$(train_csv_path) ";
	@echo "valid_csv_path=$(valid_csv_path) ";
	@echo "test_csv_path=$(test_csv_path) ";
	@echo "source_column_name=$(source_column_name) ";
	@echo "target_column_name=$(target_column_name) ";
	@echo "tokenizer_path=$(tokenizer_path) ";
	@echo "model_config_path=$(model_config_path) ";
	@echo "training_config_path=$(training_config_path) ";
	@echo "out_dir=$(out_dir) ";
	@echo "model_type=$(model_type) ";
	@python ./workflow.py \
		--train_csv_path $(train_csv_path) \
		--valid_csv_path $(valid_csv_path) \
		--test_csv_path $(test_csv_path) \
		--source_column_name $(source_column_name) \
		--target_column_name $(target_column_name) \
		--tokenizer_path $(tokenizer_path) \
		--model_config_path $(model_config_path) \
		--training_config_path $(training_config_path) \
		--out_dir $(out_dir) \
		--model_type $(model_type)