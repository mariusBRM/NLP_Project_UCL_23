from test_functions import get_results_on_csv
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.append("../models/no_fine_tuning")
from model import Net, preprocessing
import torch
import numpy as np
import pandas as pd

def test(model_path, testloader):

    # set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # list containing all the targets and outputs
    targets=[]
    outputs=[]

    # load the model
    model = Net(768,8,apply_dropout=False)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # test model
    model.eval()
    with torch.no_grad():
        for data in testloader:

            inputs, labels = data
            targets.append(labels)
            
            output = model.forward(inputs)
            
            # adding to list
            _,prediction = torch.max(output, 1)
            outputs.append(prediction)

            
    
    # get the prediction
    targets = torch.cat(targets, dim=0)
    outputs = torch.cat(outputs, dim=0)
    
    identity = np.eye(8)
    outputs = identity[outputs]
    targets = identity[targets]
    
    return outputs, targets



if __name__ == '__main__': 

    df_test = pd.read_csv("../../data/sentence_embeddings_no_fine_tuning_test.csv")
    embeddings_test = df_test["embedding"].apply(lambda x : preprocessing(x))
    labels_test = df_test["labels"].apply(lambda x : preprocessing(x))

    embeddings_test = torch.stack(embeddings_test.to_list())
    labels_test = torch.stack(labels_test.to_list())

    batch_size = 40

    test_dataset = TensorDataset(embeddings_test, labels_test)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    outputs, targets = test("../models/no_fine_tuning/classification_no_fine_tuning.pt", test_dataloader)

    get_results_on_csv("./metrics.csv", outputs, targets)


