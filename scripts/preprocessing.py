import numpy as np
import pandas as pd


# we want a dataset of the form

# text, bias type, hatespeech
def data_from_hugging_face(df):
    feature = [
        'hate_speech_score',
        'text',
        'target_race',
        'target_religion',
        'target_gender'
    ]
    df = df[feature]
    new_df = df[(df['target_race'] == True) | (df['target_religion'] == True) | (df['target_gender'] == True)]

    return new_df