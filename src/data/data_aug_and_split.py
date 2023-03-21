import pandas as pd
from sklearn.model_selection import train_test_split
import deepl

# load dataset
df = pd.read_csv("../../data/hate_speech_preprocessed.csv")
label_values = [0,1,2,3,4,5,6,7]

# remove 10k examples from "not hateful" data, to try to correct class imbalance slightly
df = df.drop(df[df['label'] == 0].sample(n=10000).index)

# split data into 80% dev, 20% test, for each class
# dev is split into 90% train, 10% val

# split dataframes per class
list_of_df_per_class = []
for l in label_values:
    list_of_df_per_class.append(df[df['label'] == l])

train_list = []
val_list= []
test_list = []

for d in list_of_df_per_class:
    dev_d, test_d = train_test_split(d, train_size=0.8, random_state=42, shuffle=True)
    train_d, val_d = train_test_split(dev_d, train_size=0.9, random_state=42, shuffle=True)
    
    train_list.append(train_d)
    val_list.append(val_d)
    test_list.append(test_d)

train_df = pd.concat(train_list, ignore_index=True).sample(frac=1)
val_df = pd.concat(val_list, ignore_index=True).sample(frac=1)
test_df = pd.concat(test_list, ignore_index=True).sample(frac=1)

# print(train_df['label'].value_counts())


# data aug: translate train data into multiple languages, depending on class frequency
auth_key = "d0fe670f-0059-2b4d-c3d5-b886c639c6fe:fx"  # Replace with your key
translator = deepl.Translator(auth_key)

aug_train_list = []
for l, d in enumerate(train_list):
    
    if l == 0:
        continue
    if l == 4:
        language = ['FR']
    if l == 1:
        language = ['FR','DE']
    if l in [5,3]:
        language = ['FR','DE','ZH']
    if l == 2:
        language = ['FR','DE','ZH','RU']
    if l in [7,6]:
        language = ['FR','DE','ZH','RU','JA']


    for i, row in d.iterrows():

        for lang in language:
            result = translator.translate_text(row['text'], target_lang=lang)
            result_back = translator.translate_text(str(result), target_lang="EN-GB")
            aug_train_list.append({'comment_id': row['comment_id'],
                                   'text': result_back,
                                   'hate_speech_score': row['hate_speech_score'],
                                   'label': row['label']})


aug_train_df = pd.DataFrame(aug_train_list)
new_train_df = pd.concat([train_df, aug_train_df], ignore_index=True).sample(frac=1)

new_train_df.to_csv("../../data/train_data.csv", index=False)
val_df.to_csv("../../data/val_data.csv", index=False)
test_df.to_csv("../../data/test_data.csv", index=False)
