import pandas as pd
import os
from datasets import load_dataset


def loader_data():
    # import dataset from Hugging face
    dataset_hugging_face = load_dataset("ucberkeley-dlab/measuring-hate-speech", 'binary')

    
    return dataset_hugging_face