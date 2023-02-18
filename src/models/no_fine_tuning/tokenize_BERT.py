import pandas as pd
from transformers import BertTokenizer
import torch
import os
import pickle

# load dataset
path_dir = os.path.abspath(os.path.join(__file__,"../../../.."))

df = pd.read_csv(path_dir + "/data/hate_speech_preprocessed.csv")

# load tokenizer
bert_model_name = "bert-base-cased"
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

max_len = max_len(df)

def tokenize_BERT(df, max_len):
    '''
    Function to return input_ids and attention masks from tokenization of BERT
    '''

    # store tokenized sentences and attention masks
    inputs_id = []
    attention_masks = []

    # to save what input corresponds to what text in case we need to make tests/check behaviour
    id_comments = df['comment_id']
    labels = torch.tensor(df['label'])


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

    return id_comments, input_ids, attention_masks, labels

id_comments, input_ids, attention_masks, labels = tokenize_BERT(df, max_len)