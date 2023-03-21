import datasets
import os
import pandas as pd
import numpy as np
from googletrans import Translator
import requests
import deepl

##################################
# Download raw hate speech dataset
##################################

# download raw dataset
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')
df = dataset['train'].to_pandas()

# set up folder and path to save it
path_dir = os.path.abspath(os.path.join(__file__,"../"))
path = os.path.join(path_dir, "data")
os.makedirs(path, exist_ok=True)

# df.to_csv(path + "/hate_speech_raw.csv", index = False)



####################
# Preprocess dataset
####################

# Note 1: Hate Speech score is the same for all annotators
# Note 2: The target of a comment was chosen to be the majority vote in the annotators
# Note 3: In case of a tie, the target is chosen at random
# Note 4: A comment is considered hate speech if its score is higher than 0.5
# Note 5: There is no "profession" target in the dataset

def preprocess_dataset(df):
    '''
    Function to preprocess hate speech dataset for classification task.
    Takes as input the raw dataframe.
    Outputs a new dataframe, with following columns:
    comment_id, text, hate_speech_score and label.

    Label has 8 possible values:
    0: comment is not hateful
    1: comment is hateful (target_race)
    2: comment is hateful (target_religion)
    3: comment is hateful (target_origin)
    4: comment is hateful (target_gender)
    5: comment is hateful (target_sexuality)
    6: comment is hateful (target_age)
    7: comment is hateful (target_disability)
    '''
    # set random seed for reproducibility
    np.random.seed(42)

    # store dataframe rows (will be concatenated)
    dataframe_list = []

    # loop through unique values of comment_id
    for id_comment in np.unique(df['comment_id']):

        # create small df with only rows corresponding to given comment
        comment_df = df.loc[df['comment_id'] == id_comment]

        # find hate speech score and text of given comment
        hs_score = comment_df['hate_speech_score'].iloc[0]
        text = comment_df['text'].iloc[0]

        # return label 0 if comment is not hateful
        if hs_score <= 0.5:
            dataframe_list.append(pd.DataFrame({'comment_id': [id_comment], 'text': [text],
                                                'hate_speech_score': [hs_score], 'label': [0]}))

        # otherwise determine the label of the comment
        else:
            # count amount of times a comment has been annotated as targetting a specific group
            list_target_counts = [comment_df['target_race'].sum(), comment_df['target_religion'].sum(),
                                  comment_df['target_origin'].sum(), comment_df['target_gender'].sum(),
                                  comment_df['target_sexuality'].sum(), comment_df['target_age'].sum(),
                                  comment_df['target_disability'].sum()]

            # find majority voting (pick at random in case of tie)
            index_max_count = np.random.choice(np.where(list_target_counts == np.max(list_target_counts))[0])
            # return label of majority voting
            dataframe_list.append(pd.DataFrame({'comment_id': [id_comment], 'text': [text],
                                                'hate_speech_score': [hs_score], 'label': [index_max_count+1]}))

    preprocessed_df = pd.concat(dataframe_list, ignore_index=True)

    return preprocessed_df

def translate_aug(df):
    # augment dataset
    auth_key = "d0fe670f-0059-2b4d-c3d5-b886c639c6fe:fx"  # Replace with your key
    translator = deepl.Translator(auth_key)
    #print(preprocessed_df['label'].value_counts())
    augmented_df = pd.DataFrame(columns=df.columns)

    for i, row in preprocessed_df[0:10].iterrows():

        if row['label'] in [1,2,3,4,5,6,7]:
            print(i)
            text = row['text']
            label = row['label']
            score = row['hate_speech_score']
            comment_id = row['comment_id']

            result = translator.translate_text(text, target_lang="FR")
            result_back = translator.translate_text(str(result) , target_lang="EN-GB")

            augmented_df = augmented_df.append({'text': result_back, 'label': label,
                                                'hate_speech_score': score,
                                                'comment_id':comment_id},
                                               ignore_index=True)

    # concatenate original and augmented datasets
    augmented_df = pd.concat([df, augmented_df], ignore_index=True)
    return augmented_df

#return concanated_set


# preprocess data and save it
preprocessed_df = preprocess_dataset(df)
augmented_set = translate_aug(preprocessed_df)

augmented_set.to_csv("data/hate_speech_augmented.csv", index=False)
preprocessed_df.to_csv("data/hate_speec_preprocessed.csv", index = False)
