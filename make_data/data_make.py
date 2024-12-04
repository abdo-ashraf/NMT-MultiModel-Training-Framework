import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import unicodedata
import contractions
from datasets import load_dataset, concatenate_datasets
import re
import os

## requirements
# !pip -q install contractions
# !pip -q install datasets

def download_data(data_type):
    assert data_type in ['both', 'opus', 'covo'], f"Bad value for data_type={data_type}, data_type should be in ['both', 'opus', 'covo']"
    datas = []
    if data_type.lower() == 'both' or data_type.lower() == 'opus':
        ds_opus = load_dataset("Helsinki-NLP/opus-100", "ar-en")
        df_opus = pd.DataFrame(ds_opus['train']['translation']).rename(columns={'en': 'source_lang', 'ar': 'target_lang',})
        datas.append(df_opus)

    if data_type.lower() == 'both' or data_type.lower() == 'covo':
        ds_covo = load_dataset("ymoslem/CoVoST2-EN-AR", "ar-en", columns=['sentence', 'translation'])
        ds_covo = concatenate_datasets([ds_covo['train'], ds_covo['validation'], ds_covo['test']])
        df_covo = pd.DataFrame(ds_covo).rename(columns={'translation': 'source_lang', 'sentence': 'target_lang',})
        datas.append(df_covo)

    return pd.concat(datas, axis=0, ignore_index=True)

# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_ar(text):
    text = unicodeToAscii(text) # Remove diacritics "التشكيل"
    text = text.replace("ة", "ه")
    text = text.replace(",", "،")
    text = text.replace(".", "۔")
    text = text.replace("?", "؟")
    text = re.sub(r"[;:]", "؛", text)
    text = re.sub(r"[^؀-ۿ0-9.!¿]+", " ", text)
    text = re.sub(r'\s+', ' ', text).strip() # Trim multiple whitespaces to one
    return text

def clean_en(text):
    text = text.lower()
    text = contractions.fix(text) # Fix contractions "it's" -> "it is"
    text = re.sub(r"[^a-z0-9?.!,¿:;'\"]+", " ", text)
    text = re.sub(r'\s+', ' ', text).strip() # Trim multiple whitespaces to one
    return text

def pre_plot(plots_dir, df_data):
    # Create a figure and two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot histogram for source_lang_length
    axes[0].hist(df_data['source_lang_length'], bins=100, color='skyblue', edgecolor='black')
    axes[0].set_title('Source Languange sentence Length')
    axes[0].set_xlabel('Length')
    axes[0].set_ylabel('Frequency')
    
    # Plot histogram for target_lang_length
    axes[1].hist(df_data['target_lang_length'], bins=100, color='salmon', edgecolor='black')
    axes[1].set_title('Target Languange sentence Length')
    axes[1].set_xlabel('Length')
    axes[1].set_ylabel('Frequency')
    
    # Save the plots
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'pre_drop_lengths.png')
    plt.savefig(plot_path, dpi=300)
    # Close the plot to prevent it from displaying
    plt.close(fig)

def drop(df_data):
    ### Personal effort (No Reference)
    indices_to_drop = []

    idx = (df_data[(df_data['target_lang_length'].isin([1])) | 
                    (df_data['source_lang_length'] == 1)]).index
    indices_to_drop += list(idx)

    idx = df_data[(df_data['source_lang_length'] > 20) | 
                    (df_data['target_lang_length'] > 20)].index
    indices_to_drop += list(idx)

    idx = (df_data[(df_data['target_lang_length'].isin([2])) & 
                (df_data['source_lang_length'] > 7)]).index
    indices_to_drop += list(idx)

    idx = (df_data[(df_data['target_lang_length'].isin([3])) & 
                (df_data['source_lang_length'] > 12)]).index
    indices_to_drop += list(idx)

    idx = (df_data[abs(df_data['target_lang_length'] - df_data['source_lang_length']) > 10]).index
    indices_to_drop += list(idx)

    indices_to_drop = list(set(indices_to_drop)) + [681448, 503657]

    filtered_data = df_data.drop(indices_to_drop).reset_index(drop=True)

    # df_data = df_data.drop_duplicates(keep='first', subset='target_lang')
    filtered_data = filtered_data.drop_duplicates(keep='first', subset='source_lang')

    filtered_data = filtered_data.replace('', pd.NA).dropna()
    filtered_data = filtered_data.replace(' ', pd.NA).dropna()
    filtered_data.isna().sum()

    return filtered_data

def post_plot(plots_dir, filtered_data):

    # Create a figure and two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Define custom tick range based on your data range
    EN_x_ticks = np.arange(0, filtered_data['source_lang_length'].max()+1, 2)
    AR_x_ticks = np.arange(0, filtered_data['target_lang_length'].max()+1, 2)

    # Plot histogram for source_lang_length
    axes[0].hist(filtered_data['source_lang_length'], bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title('Source Languange sentence Length')
    axes[0].set_xlabel('Length')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xticks(EN_x_ticks)  # Add more x-axis ticks

    # Plot histogram for target_lang_length
    axes[1].hist(filtered_data['target_lang_length'], bins=50, color='salmon', edgecolor='black')
    axes[1].set_title('Target Languange sentence Length')
    axes[1].set_xlabel('Length')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xticks(AR_x_ticks)  # Add more x-axis ticks

    # Display the plots
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'post_drop_lengths.png')
    plt.savefig(plot_path, dpi=300)
    # Close the plot to prevent it from displaying
    plt.close(fig)

def make_data(out_dir, data_type='both', valid_test_split=0.4, seed=123):

    data_out_dir = os.path.join(out_dir, 'data')
    os.makedirs(plots_out_dir, exist_ok=True)
    plots_out_dir = os.path.join(out_dir, 'plots')
    os.makedirs(data_out_dir, exist_ok=True)

    df_data = download_data(data_type)
    df_data['source_lang'] = df_data['source_lang'].apply(clean_en)
    df_data['target_lang'] = df_data['target_lang'].apply(clean_ar)
    df_data['source_lang_length'] = df_data['source_lang'].apply(lambda x: len(x.split(' ')))
    df_data['target_lang_length'] = df_data['target_lang'].apply(lambda x: len(x.split(' ')))
    if os.path.exists(str(plots_out_dir)):
        pre_plot(plots_out_dir, df_data)

    filtered_data = drop(df_data)

    if os.path.exists(str(plots_out_dir)):
        post_plot(plots_out_dir, filtered_data)

    df_train, df_test = train_test_split(filtered_data, test_size=valid_test_split, shuffle=True, random_state=seed)
    df_valid, df_test = train_test_split(df_test, test_size=0.5, shuffle=True, random_state=seed)
    print(df_train.shape, df_test.shape, df_valid.shape, sep=', ')

    df_train.to_csv(os.path.join(data_out_dir,'df_train.csv'), index=False)
    df_valid.to_csv(os.path.join(data_out_dir,'df_valid.csv'), index=False)
    df_test.to_csv(os.path.join(data_out_dir,'df_test.csv'), index=False)

    return df_train, df_valid, df_test



