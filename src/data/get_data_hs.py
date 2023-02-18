import datasets
import os
import pandas as pd
import numpy as np

##################################
# Download raw hate speech dataset
##################################

# download raw dataset
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')   
df = dataset['train'].to_pandas()

# set up folder and path to save it
path_dir = os.path.abspath(os.path.join(__file__,"../../.."))
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
            
    return pd.concat(dataframe_list, ignore_index=True)


# preprocess data and save it
preprocessed_df = preprocess_dataset(df)
preprocessed_df.to_csv(path + "/hate_speech_preprocessed.csv", index = False)