# Toxic Comment Classification
## Overview
Every online platform which has an open forum faces an issue of people posting inappropriate comments, which if uncontrolled, can lead to loss of users, reputation and revenue. However, it is impractical and expensive for humans to keep track of all the messages other people post.
This project aims to develop a a multi-label classifier, to assign forum posts to one or more of the 6 classes.


## Dataset
Dataset is from [Kaggle datasets](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

## Python Libraries
Code was done on Python 3.11.9. Packages can be found in requirements.txt


## Findings
* Exploratory Data Analysis (EDA): Dataset is made of 6 classes

    - Majority classes had 10% of labels had classes assigned to them. Minority classes had up to 4-10%, minority classes had up to 1% of all cases.

* Models: 
    - Roberta as pretrained Transfer Learning backbone model. AdamW and RMStop optimizers were tested. The best model was fined tuned with AdamW, weighted BCE, which predicted majority classes up to 59-66% F1 score.

    - Looking into bad predictions, comments with aggressive/passive aggressive sentiment were classified into either of 6 classes.



