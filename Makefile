needed_data ?= 1  # Default to 1 if needed_data isn't passed
OUT_DIR ?= ./out/  # Default to ./out/ if OUT_DIR isn't passed
SEED ?= 123
VALID_TEST_SPLIT ?= 0.1
maxlen ?= 25

.PHONY: data
.PHONY: tokenizer
.PHONY: model

setup:
	pip install -r requirements.txt

data:
	@if [ "$(needed_data)" = "1" ]; then \
	echo "Making ar_en at $(OUT_DIR)data/";\
	    python ./make_data/ar_en_data_workflow.py \
	        --out_dir $(OUT_DIR) \
	        --maxlen $(maxlen) \
	        --seed $(SEED) \
	        --valid_test_split $(VALID_TEST_SPLIT); \
	elif [ "$(needed_data)" = "2" ]; then \
	echo "Making sp-en at $(OUT_DIR)data/";\
	    python ./make_data/sp_en_data_workflow.py \
	        --out_dir $(OUT_DIR) \
	        --seed $(SEED) \
	        --valid_test_split $(VALID_TEST_SPLIT); \
	else \
	    echo "Invalid data type. Choose from: 1 (ar-en), 2 (sp-en)"; \
	    exit 1; \
	fi

tokenizer:
	echo "Making tokenizer at";
