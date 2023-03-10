import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def preprocessing(embedding):
    """
    Given one embeddings in input, the function apply preprocessing steps. 
    The function returns a clean embedding of type torch.tensor
    """

    l = embedding.replace('\n','') #remove '\n
    l = l.replace(' ','') #remove any space
    l = l.replace(',grad_fn=<SelectBackward0>','') #remove grad_fn=<SelectBackward0>
    l1 = l[7:] #remove tensor(
    l2 = l1[:-1] #remove )
    return torch.tensor(eval(l2)) #get the list inside the string and convert it into torch.tensor



class Net(nn.Module):
    def __init__(self,len_input_layer, len_output_layer, apply_dropout=False):
        super().__init__()
        
        self.fc1 = nn.Linear(len_input_layer,568)
        self.fc2 = nn.Linear(568, 366)
        self.fc3 = nn.Linear(366, 124)
        self.fc4 = nn.Linear(124, len_output_layer)
        self.apply_dropout = apply_dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.apply_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.apply_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.apply_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc4(x))
        return x
    


def compute_metrics(dataloader, model, criterion):
    """
    
    This function returns the accuracy of the model on the data given in inputs
    """

    list_labels = []
    list_pred = []

    loss = 0.0
    for (inputs, labels) in dataloader:
        outputs = model.forward(inputs)
        loss += criterion(outputs, labels).item()
        _,predicted = torch.max(outputs, 1)
        list_labels.append(labels)
        list_pred.append(predicted)

    y_true = torch.cat(list_labels).numpy()
    y_pred = torch.cat(list_pred).numpy()
    
    loss = loss / len(dataloader)
    acc = accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    return loss, acc, macro_f1



def training_model(nb_epochs, train_dataloader, val_dataloader, model, optimizer, criterion, patience):
    """
    This function trains the model on training data
    """

    for epoch in range(nb_epochs):

        model.train()
        running_loss = 0
        for j, data in enumerate(train_dataloader, 0):

            input, labels = data
            
            outputs = model.forward(input)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  

            running_loss += loss.item()
            
            print('\rEpoch: {}\tbatch: {}\tLoss =  {:.3f}'.format(epoch, j, loss), end="")

        running_loss = running_loss / len(train_dataloader)
        
        # validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_macro_f1 = compute_metrics(dataloader=val_dataloader,network=model,criterion=criterion)
            print("Epoch {}: train CE loss = {:.5f}".format(epoch+1, running_loss),
                      '|| Valid: CE loss = {:.5f}   acc = {:.2f}%   macro-F1 = {:.4f}'.format(val_loss, val_acc, val_macro_f1))

        # early-stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            dict_model = model.state_dict()
            pat = 0
        else:
            pat += 1
            if pat == patience:
                print("Early Stopping: Validation Loss did not decrease for", patience, "epochs.")
                break

    torch.save(dict_model, 'classification_no_fine_tuning.pt')



if __name__ == '__main__':

    #Preprocessing of the data
    df = pd.read_csv("../../../data/hs_sentence_embeddings_no_fine_tuning.csv")
    all_embeddings = df["embedding"].apply(lambda x : preprocessing(x)) 
    all_labels = df["labels"].apply(lambda x : preprocessing(x)) 
    assert all_embeddings.shape[0] == all_labels.shape[0]

    all_embeddings = all_embeddings.to_list()
    all_labels = all_labels.to_list()

    batch_size = 80

    split = int(df.shape[0]*0.8)

    #development data correspond to 80% of the original dataset
    dev_data = all_embeddings[:split]
    dev_labels = all_labels[:split]
    assert len(dev_data) == len(dev_labels)

    split_dev = int(len(dev_data)*0.9)
    # train data is 90% of development data
    train_data = torch.stack(dev_data[:split_dev])
    train_labels = torch.stack(dev_labels[:split_dev])
    assert len(train_data) == len(train_labels)
    train_dataset = TensorDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    # validation data is 10% of development data
    val_data = torch.stack(dev_data[split_dev:])
    val_labels = torch.stack(dev_labels[split_dev:])
    assert len(val_data) == len(val_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

    #test data correspond to 20% of the original dataset
    test_data = torch.stack(all_embeddings[split:])
    test_labels = torch.stack(all_labels[split:])
    assert len(test_data) == len(test_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)


    ## training

    nb_epochs = 500
    patience = 10
    model = Net(768,8,apply_dropout=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # optim.Adam(model.parameters(), lr=0.001)
    # optim.AdamW(model.parameters(), lr=0.001)
    

    training_model(nb_epochs, train_dataloader, val_dataloader, model, optimizer, criterion, patience)

    # testing
    print("\n--------------------------------------------------------------------------------------\n")
    train_loss, train_acc, train_macro_f1 = compute_metrics(train_dataloader,model,criterion)
    print("Metrics on training data : CE loss = {:.5f} | acc = {:.2f}% | macro-F1 = {:.4f}".format(train_loss, train_acc, train_macro_f1),"\n")
    val_loss, val_acc, val_macro_f1 = compute_metrics(val_dataloader,model,criterion)
    print("Metrics on validation data : CE loss = {:.5f} | acc = {:.2f}% | macro-F1 = {:.4f}".format(val_loss, val_acc, val_macro_f1),"\n")
    test_loss, test_acc, test_macro_f1 = compute_metrics(test_dataloader,model,criterion)
    print("Metrics on test data : CE loss = {:.5f} | acc = {:.2f}% | macro-F1 = {:.4f}".format(test_loss, test_acc, test_macro_f1),"\n")
    print("--------------------------------------------------------------------------------------\n")


