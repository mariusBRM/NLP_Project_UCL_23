import pandas as pd
from transformers import BertTokenizer
import torch
import os
import pickle

# load tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

def max_len(df):
    '''
    Function to find maximal length of sentence
    '''

    max_len = 0

    for comment in df['text']:
        inputs_id = tokenizer.encode(comment,add_special_tokens=True)
        max_len = max(max_len, len(inputs_id))

    return max_len




def get_inputs_ids_attention_masks(df, max_len):
    
    inputs_id = []
    attention_masks = []

    for comment in df['text']:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(comment,
                                             add_special_tokens = True, #Add CLS and SEP tokens
                                             max_length = max_len, 
                                             padding = 'max_length', #pad_to_max_length = True
                                             return_attention_mask = True,
                                             return_tensors = 'pt')
    
        inputs_id.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask']) # differentiates padding from non-padding tokens

    # Convert to tensor
    input_ids = torch.cat(inputs_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks



def tokenize_BERT():
    '''
    Function to return input_ids and attention masks from tokenization of BERT
    '''

    # load datasets
    df_train = pd.read_csv("../../../data/train_data.csv")
    df_val = pd.read_csv("../../../data/val_data.csv")
    df_test = pd.read_csv("../../../data/test_data.csv")


    #get max len
    max_len_train = max_len(df_train)
    max_len_val = max_len(df_val)
    max_len_test = max_len(df_test)

    max_len_ = max([max_len_train, max_len_val, max_len_test])



    # to save what input corresponds to what text in case we need to make tests/check behaviour
    id_comments_train = df_train['comment_id']
    labels_train = torch.tensor(df_train['label'])

    id_comments_val = df_val['comment_id']
    labels_val = torch.tensor(df_val['label'])

    id_comments_test = df_test['comment_id']
    labels_test = torch.tensor(df_test['label'])

    # store tokenized sentences and attention masks
    inputs_id_train, attention_masks_train = get_inputs_ids_attention_masks(df_train, max_len_)

    inputs_id_val, attention_masks_val = get_inputs_ids_attention_masks(df_val, max_len_)
    
    inputs_id_test, attention_masks_test = get_inputs_ids_attention_masks(df_test,max_len_)



    train_data = (id_comments_train,inputs_id_train,attention_masks_train,labels_train)
    val_data = (id_comments_val,inputs_id_val,attention_masks_val,labels_val)
    test_data = (id_comments_test,inputs_id_test,attention_masks_test,labels_test)

    return train_data, val_data, test_data
