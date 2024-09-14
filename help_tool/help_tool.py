import os
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import inspect
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pickle
import torch
import nltk
# nltk.download('stopwords')
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

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


def plot_per_class(num_labels, train_loss_per_class, val_loss_per_class, train_accuracy_per_class, val_accuracy_per_class):

    # Plot accuracy and loss for each class
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Training Loss per class
    for i in range(num_labels):
        axs[0, 0].plot(train_loss_per_class[i], label=f'Class {i+1}')
    axs[0, 0].set_title('Training Loss per Class')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Validation Loss per class
    for i in range(num_labels):
        axs[0, 1].plot(val_loss_per_class[i], label=f'Class {i+1}')
    axs[0, 1].set_title('Validation Loss per Class')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()

    # Training Accuracy per class
    for i in range(num_labels):
        axs[1, 0].plot(train_accuracy_per_class[i], label=f'Class {i+1}')
    axs[1, 0].set_title('Training Accuracy per Class')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy (%)')
    axs[1, 0].legend()

    # Validation Accuracy per class
    for i in range(num_labels):
        axs[1, 1].plot(val_accuracy_per_class[i], label=f'Class {i+1}')
    axs[1, 1].set_title('Validation Accuracy per Class')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy (%)')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()



def predictions(model, device, test_dataloader):
    model.eval()

    #track variables
    logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

    # Predict
    for i, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            # Forward pass
            outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)

            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)

    # Flatten outputs
    tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    # Converting flattened binary values to boolean values
    true_bools = [tl==1 for tl in true_labels]

    return(tokenized_texts, pred_labels, true_bools)


def altered_tresshold_classification_report(model, device, test_dataloader, test_label_cols, threshold):
    (tokenized_texts, pred_labels, true_bools) = predictions(model, device, test_dataloader)


    pred_bools = [pl>threshold for pl in pred_labels] #boolean output after thresholding

    print('F1: ', f"{f1_score(true_bools, pred_bools,average='micro'):.2f}")
    print('Accuracy: ', f'{accuracy_score(true_bools, pred_bools):.2f}','\n')

    print(classification_report(true_bools,pred_bools,target_names=test_label_cols, zero_division=0))



def optimize_threshold(pred_labels, true_bools, label_cols):

    # Define macro thresholds (0.1, 0.2, ..., 0.9)
    macro_thresholds = np.array(range(1, 10)) / 10

    f1_results, flat_acc_results = [], []
    
    # Iterate over macro thresholds to find the best threshold for F1 score
    for th in macro_thresholds:
        pred_bools = [pl > th for pl in pred_labels]
        
        # Calculate F1 score (micro) and accuracy, with zero_division=0 to handle undefined precision cases
        test_f1_accuracy = f1_score(true_bools, pred_bools, average='micro', zero_division=0)
        test_flat_accuracy = accuracy_score(true_bools, pred_bools)
        
        f1_results.append(test_f1_accuracy)
        flat_acc_results.append(test_flat_accuracy)

    # Find the best macro threshold based on F1 score
    best_macro_th = macro_thresholds[np.argmax(f1_results)]

    # Define micro thresholds (fine-tuning around the best macro threshold)
    micro_thresholds = (np.array(range(10)) / 100) + best_macro_th

    f1_results, flat_acc_results = [], []
    
    # Iterate over micro thresholds for finer optimization
    for th in micro_thresholds:
        pred_bools = [pl > th for pl in pred_labels]
        test_f1_accuracy = f1_score(true_bools, pred_bools, average='micro', zero_division=0)
        test_flat_accuracy = accuracy_score(true_bools, pred_bools)
        
        f1_results.append(test_f1_accuracy)
        flat_acc_results.append(test_flat_accuracy)

    # Find the best micro threshold based on F1 score
    best_f1_idx = np.argmax(f1_results)

    # Print the best threshold and corresponding scores
    print('Best Threshold: ', f'{micro_thresholds[best_f1_idx]:.2f}')
    print('F1: ', f'{f1_results[best_f1_idx]:.2f}')
    print('Accuracy: ', f'{flat_acc_results[best_f1_idx]:.2f}', '\n')

    # Apply the best threshold to predictions
    best_pred_bools = [pl > micro_thresholds[best_f1_idx] for pl in pred_labels]
    
    # Generate the classification report, again using zero_division=0 to avoid warnings
    print(classification_report(true_bools, best_pred_bools, target_names=label_cols, zero_division=0))


def random_sampler_dataloader_creation(batch_size, inputs, labels, masks):
    # Convert all of our data into torch tensors, the required datatype for our model
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    masks = torch.tensor(masks)

    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader



def sequential_sampler_dataloader_creation(batch_size, inputs, labels, masks):
    # Convert all of our data into torch tensors, the required datatype for our model
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    masks = torch.tensor(masks)

    data = TensorDataset(inputs, masks, labels)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader