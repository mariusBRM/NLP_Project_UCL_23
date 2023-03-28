import torch
import numpy as np
from test_functions import get_results_on_csv
from tokenize_BERT import train_data, val_data, test_data

sys.path.append("../models/fine_tuning")
from model import BERTForFineTuningtWithPooling, get_class
from torch.utils.data import TensorDataset, DataLoader


def test(model_path, testloader):

    # set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # list containing all the targets and outputs
    targets=[]
    outputs=[]

    # load the model
    model = BERTForFineTuningtWithPooling()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # test model
    model.eval()
    with torch.no_grad():
        for data in testloader:

            #ids = data['input_ids'].to(device, dtype = torch.long)
            #attention_mask = data['attention_mask'].to(device, dtype = torch.long)
            #labels = data['labels'].to(device, dtype = torch.float)

            ids = data[0].to(device, dtype = torch.long)
            attention_mask = data[1].to(device, dtype = torch.long)
            labels = data[2].to(device, dtype = torch.long)

            _, output = model.forward(ids, attention_mask)
            # adding to list
            targets.extend(labels.cpu().detach().numpy().tolist())
            outputs.extend(output.cpu().detach().numpy().tolist())
    
    # get the prediction
    outputs = get_class(outputs)
    outputs = np.array(outputs)
    targets = np.array(targets)
    
    return outputs, targets

if __name__ == '__main__': 

    

    batch_size = 80

    batch_size = 80

    test_dataset = TensorDataset(test_data[1], test_data[2], test_data[3])
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)


    outputs, targets = test("../models/no_fine_tuning/classification_no_fine_tuning.pt", test_dataloader)

    get_results_on_csv("./metrics_fine_tuning.csv", outputs, targets)


