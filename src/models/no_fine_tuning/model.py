import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler

def preprocessing(embedding):
    """
    Given one embeddings in input, the function apply preprocessing steps. 
    The function returns a clean embedding of type torch.tensor
    """

    l = embedding.replace('\n','') #remove '\n
    l = l.replace(' ','') #remove any space
    l = l.replace(',grad_fn=<SelectBackward0>','') #remove grad_fn=<SelectBackward0>
    l = l.replace(",device='cuda:0'",'')
    l1 = l[7:] #remove tensor(
    l2 = l1[:-1] #remove )
    return torch.tensor(eval(l2)) #get the list inside the string and convert it into torch.tensor



class Net(nn.Module):
    def __init__(self,len_input_layer, len_output_layer, apply_dropout=False):
        super().__init__()
        
        self.fc1 = nn.Linear(len_input_layer,684)
        self.fc2 = nn.Linear(684, 536)
        self.fc3 = nn.Linear(536, 498)
        self.fc4 = nn.Linear(498, 124)
        self.fc5 = nn.Linear(124,len_output_layer)
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
        if self.apply_dropout:
            x = self.dropout(x)
        
        x = self.fc5(x)
        
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

    

    best_val_loss = np.inf

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
        print("\n")
        # validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_macro_f1 = compute_metrics(val_dataloader,model,criterion)
            print("Epoch {}: train CE loss = {:.5f}".format(epoch+1, running_loss),
                      '|| Valid: CE loss = {:.5f}   acc = {:.2f}%   macro-F1 = {:.4f}'.format(val_loss, val_acc, val_macro_f1))

        # early-stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            dict_model = model.state_dict()
            pat = 0
        else:
            pat += 1
            print("pat ", pat)
            if pat == patience:
                print("Early Stopping: Validation Loss did not decrease for", patience, "epochs.")
                break

        
        
        print("\n")
    torch.save(dict_model, 'classification_no_fine_tuning.pt')



if __name__ == '__main__':

    ##collect data
    df_train = pd.read_csv("../../../data/sentence_embeddings_no_fine_tuning_train.csv")
    embeddings_train = df_train["embedding"].apply(lambda x : preprocessing(x))
    labels_train = df_train["labels"].apply(lambda x : preprocessing(x))

    df_val = pd.read_csv("../../../data/sentence_embeddings_no_fine_tuning_val.csv")
    embeddings_val = df_val["embedding"].apply(lambda x : preprocessing(x))
    labels_val = df_val["labels"].apply(lambda x : preprocessing(x))

    df_test = pd.read_csv("../../../data/sentence_embeddings_no_fine_tuning_test.csv")
    embeddings_test = df_test["embedding"].apply(lambda x : preprocessing(x))
    labels_test = df_test["labels"].apply(lambda x : preprocessing(x))

    batch_size = 40

    embeddings_train = torch.stack(embeddings_train.to_list())
    labels_train = torch.stack(labels_train.to_list())

    embeddings_val = torch.stack(embeddings_val.to_list())
    labels_val = torch.stack(labels_val.to_list())

    embeddings_test = torch.stack(embeddings_test.to_list())
    labels_test = torch.stack(labels_test.to_list())


    train_dataset = TensorDataset(embeddings_train, labels_train)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    val_dataset = TensorDataset(embeddings_val, labels_val)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    test_dataset = TensorDataset(embeddings_test, labels_test)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)


    ## training

    nb_epochs = 1000
    patience = 35
    model = Net(768,8,apply_dropout=False)
    
    weights = 1 / (torch.sqrt(torch.unique(labels_train, return_counts = True)[1]))
    criterion = nn.CrossEntropyLoss(weight = weights)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    training_model(nb_epochs, train_dataloader, val_dataloader, model, optimizer, criterion, patience)


    ## testing
    print("\n--------------------------------------------------------------------------------------\n")
    train_loss, train_acc, train_macro_f1 = compute_metrics(train_dataloader,model,criterion)
    print("Metrics on training data : CE loss = {:.5f} | acc = {:.2f}% | macro-F1 = {:.4f}".format(train_loss, train_acc, train_macro_f1),"\n")
    val_loss, val_acc, val_macro_f1 = compute_metrics(val_dataloader,model,criterion)
    print("Metrics on validation data : CE loss = {:.5f} | acc = {:.2f}% | macro-F1 = {:.4f}".format(val_loss, val_acc, val_macro_f1),"\n")
    test_loss, test_acc, test_macro_f1 = compute_metrics(test_dataloader,model,criterion)
    print("Metrics on test data : CE loss = {:.5f} | acc = {:.2f}% | macro-F1 = {:.4f}".format(test_loss, test_acc, test_macro_f1),"\n")
    print("--------------------------------------------------------------------------------------\n")


