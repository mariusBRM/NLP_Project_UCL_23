import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import ast

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
    def __init__(self,len_input_layer, len_output_layer):
        super().__init__()
        
        self.fc1 = nn.Linear(len_input_layer,568)
        self.fc2 = nn.Linear(568, 366)
        self.fc3 = nn.Linear(366, 124)
        self.fc4 = nn.Linear(124, len_output_layer)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    

def training_model(nb_epochs,train_dataloader,model,optimizer,loss):
    """
    This function trains the model on training data
    """
    list_running_loss = []

    for epoch in range(nb_epochs):
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


        print("\nRunning_loss = ", running_loss,)
        
        #if loss from previous epoch and current loss are very close => we have converged
        if epoch >=1 and np.abs(list_running_loss[-1] - running_loss) < 0.001:
            break

        list_running_loss.append(running_loss)

        print("")


def compute_accuracy(Dataloader, model):
    """
    
    This function returns the accuracy of the model on tha data given in inputs
    """

    nb_good_prediction = 0
    total=0

    for i,data in enumerate(Dataloader,0):
        inputs, labels = data
        outputs = model.forward(inputs)
        _,predicted = torch.max(outputs, 1)
        nb_good_prediction += predicted.eq(labels.data).cpu().sum()
        total += labels.size(0)  
        
    return nb_good_prediction / total 




if __name__ == '__main__':

    #Preprocessing of the data
    df = pd.read_csv("../../../data/hs_sentence_embeddings_no_fine_tuning.csv")
    all_embeddings = df["embedding"].apply(lambda x : preprocessing(x)) 
    all_labels = df["labels"].apply(lambda x : preprocessing(x)) 
    assert all_embeddings.shape[0] == all_labels.shape[0]




    split = int(df.shape[0]*0.7)

    #training data correspond to 70% of the original dataset
    training_data = torch.stack(all_embeddings[:split].to_list())
    training_labels = torch.stack(all_labels[:split].to_list())
    assert len(training_data) == len(training_labels)

    #test data correspond to 30% of the original dataset
    test_data = torch.stack(all_embeddings[split:].to_list())
    test_labels = torch.stack(all_labels[split:].to_list())
    assert len(test_data) == len(test_labels)


    batch_size = 80
    train_data = training_data
    train_labels = training_labels
    nb_epochs = 500

    model = Net(768,8)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #training data
    my_dataset = TensorDataset(train_data, train_labels)
    train_dataloader = DataLoader(my_dataset, batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    training_model(nb_epochs, train_dataloader, model,optimizer,criterion)

    #testing data
    my_dataset = TensorDataset(test_data, test_labels)
    test_dataloader = DataLoader(my_dataset, batch_size, shuffle=True)

    #Accuracy 
    print("\n--------------------------------------------------------------------------------------\n")
    print("Accuracy on training data : ", compute_accuracy(train_dataloader,model),"\n")
    print("Accuracy on test data : ", compute_accuracy(test_dataloader,model),"\n")
    print("--------------------------------------------------------------------------------------\n")


