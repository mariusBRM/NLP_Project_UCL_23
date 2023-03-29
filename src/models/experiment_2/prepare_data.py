import pandas as pd
from sklearn.model_selection import train_test_split

# load datasets
df_original_data = pd.read_csv("../../../data/hate_speech_preprocessed.csv")
df_train = pd.read_csv("../../../data/train_data.csv")
df_val = pd.read_csv("../../../data/val_data.csv")
df_test = pd.read_csv("../../../data/test_data.csv")

#re-add 10000 datapoints cut for previous experiment
# get 0 labels from original dataset and remove it from train/val/test splits
df_original_data = df_original_data[df_original_data.label == 0]
df_train = df_train[df_train.label != 0]
df_val = df_val[df_val.label != 0]
df_test = df_test[df_test.label != 0]

# resplit original 0 labelled data
dev_d, test_d = train_test_split(df_original_data, train_size=0.8, random_state=42, shuffle=True)
train_d, val_d = train_test_split(dev_d, train_size=0.9, random_state=42, shuffle=True)

# reinsert 0 labelled data into train/val/test splits
df_train = pd.concat([df_train, train_d], ignore_index=True).sample(frac=1)
df_val = pd.concat([df_val, val_d], ignore_index=True).sample(frac=1)
df_test = pd.concat([df_test, test_d], ignore_index=True).sample(frac=1)

