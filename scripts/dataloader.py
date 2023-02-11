import pandas as pd
import os
from datasets import load_dataset


def loader_data():
    # import dataset from Hugging face
    dataset_hugging_face = load_dataset("ucberkeley-dlab/measuring-hate-speech", 'binary')

    # current_directory = os.getcwd()
    # os.chdir("..")
    # file_data = os.getcwd() + '\data'
    # #import dataset from convabus
    # dataset_convabus = pd.read_csv(file_data + '\convabuse.csv', on_bad_lines='skip')
    # # import dataset from 
    # dataset_DGHS = pd.read_csv(file_data + '\Dynamically Generated Hate Dataset v0.2.2.csv', on_bad_lines='skip')
    
    return dataset_hugging_face