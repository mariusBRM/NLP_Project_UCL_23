import sys
import train
import preprocessing
import torch
import dataloader

def main():
    dataset = dataloader.loader_data()
    df_train = dataset['train'].to_pandas()
    df = preprocessing.data_from_hugging_face(df_train)
    
    X = df['text']
    y = df[['hate_speech_score', 'target_race', 'target_religion', 'target_gender']]
    y[['target_race', 'target_religion', 'target_gender']] = y[['target_race', 'target_religion', 'target_gender']].astype(int)

    print(X.head())
    return None

if __name__=="__main__":
    sys.exit(main())