import os
import pandas as pd
import re
from urllib.parse import urlparse

import inspect

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords


pd.plotting.register_matplotlib_converters()

"""Statistics"""
alpha = 0.05  # Significance level
confidence_level = 0.95

cmap='rocket'

def csv_download(relative_path: str) -> pd.DataFrame:
    """Download data."""
    absolute_path = os.path.abspath(relative_path)
    df = pd.read_csv(absolute_path, index_col=False, header=0)

    return df


def first_look(df: pd.DataFrame) -> None:
    """Performs initial data set analysis."""
    df_size = df.shape

    df_type = df.dtypes.to_frame().T.rename(index={df.index[0]: 'dtypes'})
    df_null = df.apply(lambda x: x.isna().sum()).to_frame().T.rename(
        index={df.index[0]: 'Null values, Count'})

    df_null_proc = round(df_null / df_size[0] * 100, 1)
    df_null_proc = df_null_proc.rename(
        index={df_null.index[0]: 'Null values, %'})

    info_df = pd.concat([df_type, df_null, df_null_proc])

    print(f'Dataset has {df.shape[0]} observations and {df_size[1]} features')
    print(
        f'Columns with all empty values {df.columns[df.isna().all(axis=0)].tolist()}')
    print(f'Dataset has {df.duplicated().sum()} duplicates')

    return info_df.T



def replace_url_with_domain(text):
    # Regular expression to find URLs
    url_pattern = r'https?://[^\s]+'
    
    def extract_domain(url):
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Remove 'www.' if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            # Split domain and remove common TLDs
            domain_parts = domain.split('.')
            if len(domain_parts) > 1:
                # Remove the last part (TLD)
                domain = '.'.join(domain_parts[:-1])
                
            return domain
        except Exception as e:
            return url
    
    # Find all URLs in the text
    urls = re.findall(url_pattern, text)
    
    # Replace each URL with its domain name
    for url in urls:
        domain = extract_domain(url)
        text = text.replace(url, domain)
    
    return text


def replace_filenames(text):

    def replace_match(match):
        return match.group(1)
    
    # Replace the filenames in the text using re.sub
    replaced_text = re.sub(r'File:([A-Za-z0-9_]+)\.(jpg|jpeg|png|gif)', replace_match, text)
    replaced_text = re.sub(r'([A-Za-z0-9_]+)\.(jpg|jpeg|png|gif)', replace_match, text)
    
    return replaced_text



def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text




def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def remove_non_ascii(text):
    return ''.join([char for char in text if ord(char) <= 127])


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


def model_dataset_distribution(original_df, X_train, X_val#, X_test
                               ):
    df  = pd.DataFrame()
    for i in [X_train, X_val#, X_test
              ]:
        df1 = (original_df[original_df['id'].isin(i['id'])].drop(columns=['id', 'comment_text']).sum() / original_df[original_df['id'].isin(i['id'])].shape[0] * 100).round(1).to_frame().reset_index().rename(columns={'index': 'Type', 0: 'Proc'})
        df1['Dataset'] = retrieve_name(i)
        df = pd.concat([df1, df], ignore_index=True, sort=False)
    
    return df