import keras
import pathlib
import string
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os


def sp_en_data(out_dir, valid_test_split, seed):
    os.makedirs(out_dir, exist_ok=True)

    text_file = keras.utils.get_file(
        fname="spa-eng.zip",
        origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
        extract=True)
    
    text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"

    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")
    def custom_standardization(input_string):
        lowercase = input_string.lower()
        return re.sub("[%s]" % re.escape(strip_chars), "", lowercase)

    with open(text_file) as f:
        lines = f.read().split("\n")[:-1]
    text_pairs = []
    for line in lines:
        eng, spa = line.split("\t")
        text_pairs.append((eng.lower(), custom_standardization(spa)))


    train_pairs, test_pairs = train_test_split(text_pairs, test_size=valid_test_split, shuffle=True, random_state=seed)
    val_pairs, test_pairs = train_test_split(test_pairs, test_size=0.5, shuffle=True, random_state=seed)

    df_train = pd.DataFrame(train_pairs, columns=['en', 'sp'])
    df_valid = pd.DataFrame(val_pairs, columns=['en', 'sp'])
    df_test = pd.DataFrame(test_pairs, columns=['en', 'sp'])
    print(df_train.shape, df_test.shape, df_valid.shape, sep=', ')
    df_train.to_csv(os.path.join(out_dir, 'en-sp_train.csv'))
    df_valid.to_csv(os.path.join(out_dir, 'en-sp_valid.csv'))
    df_test.to_csv(os.path.join(out_dir, 'en-sp_test.csv'))

    return df_train, df_valid, df_test