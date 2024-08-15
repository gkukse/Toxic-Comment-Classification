import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score
                             )


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


def distribution_check(df: pd.DataFrame) -> None:
    """Box plot graph for identifying numeric column outliers, normality of distribution."""
    df = df.reset_index(drop=True)

    for feature in df.columns:

        if df[feature].dtype.name in ['object', 'bool']:
            pass

        else:

            fig, axes = plt.subplots(1, 3, figsize=(12, 3))

            print(f'{feature}')

            df.boxplot(column=feature, ax=axes[0])
            axes[0].set_title(
                f'{feature} ranges from {df[feature].min()} to {df[feature].max()}')

            sns.histplot(data=df, x=feature, kde=True, bins=20, ax=axes[1])
            axes[1].set_title(f'Distribution of {feature}')

            sm.qqplot(df[feature].dropna(), line='s', ax=axes[2])
            axes[2].set_title(f'Q-Q plot of {feature}')

            plt.tight_layout()
            plt.show()


def contains_non_english(text):
    for char in text:
        if ord(char) > 127:  # Non-ASCII character
            return True
    return False