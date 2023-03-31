import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from sklearn.metrics import accuracy_score,f1_score
import pickle
from tokenize_BERT import tokenize_BERT
import torch.nn as nn

from prepare_data import *



def get_class(output):
    classes = []
    for score in output:
      if score > 0.5:
        classes.append(torch.tensor(1))
      else:
        classes.append(torch.tensor(0))

    return torch.tensor(classes)






# Custom the data for our need
class HateSpeechData(Dataset):
    def __init__(self, X):
        self.X = (X[1], X[2])
        self.y = X[3]
        self.id = X[0]
        
    def __getitem__(self, index):
        # get the item out of the tuple
        inputs_id = self.X[0][index]
        attention_mask = self.X[1][index]
        label = self.y[index]
        id = self.id[index]
        
        if label !=0:
            label = torch.tensor(1)

        # create dictionnary
        item = {
            'input_ids':inputs_id,
            'attention_mask':attention_mask,
            'labels':label,
            'comment_id': id
        }
        return item
    
    def __len__(self):
        return len(self.X[1])




# Dataloader
def data_loader(data,batch_size):
    

    # Map style for Dataloader
    dataset = HateSpeechData(data)

    # dataloader
    dataloader_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return dataloader_loader




class BERTForFineTuningtWithPooling(torch.nn.Module):
    def __init__(self):
        super(BERTForFineTuningtWithPooling, self).__init__()
        # first layer is the bert
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        # apply a dropout
        self.l2 = torch.nn.Dropout(0.1)
        self.l3 = torch.nn.Linear(768, 1)
    
    def forward(self, ids, mask):
        outputs = self.l1(ids, attention_mask=mask)
        pooled_output = outputs[1] 
        output_2 = self.l2(pooled_output)
        output = self.l3(output_2)
        output = torch.sigmoid(output)
        #return outputs.hidden_states, output
        return torch.squeeze(output)


def loss_fn(outputs, targets):
    
    loss = nn.BCELoss()
    return loss(outputs, targets)

def validation(validation_loader, model):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    fin_targets=[]
    fin_outputs=[]
    running_loss = 0.0

    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):

            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.float)
            
            # forward
            output = model.forward(ids, mask)
            # evaluate the loss
            loss = loss_fn(output, targets)

            # adding to list
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(output.cpu().detach().numpy().tolist())

            # add the loss to the running loss
            running_loss+=loss.item()

    return fin_outputs, fin_targets, running_loss/len(validation_loader)


def training_model(nb_epochs, train_dataloader, val_dataloader, patience):
    """
    This function trains the model on training data
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BERTForFineTuningtWithPooling()
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    best_val_loss = np.inf
    # keep track of the performances
    summary = []


    for epoch in range(nb_epochs):
            # dict containing the information
        report_epoch = {
                'epoch': epoch+1,
                'training_loss': 0.0,
                'valid_loss':0.0,
                'valid_accuracy':0.0,
                'valid_f1_macro':0.0
            }
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):

            ids = data['input_ids'].to(device, dtype = torch.long)
            attention_mask = data['attention_mask'].to(device, dtype = torch.long)
            labels = data['labels'].to(device, dtype = torch.float)
            
             # initialize the optimizer
            optimizer.zero_grad()

            #forward inputs
            output = model.forward(ids, attention_mask)
            
            # define the loss
            loss = loss_fn(output, labels)

            # backpropagate
            loss.backward()
            
            optimizer.step()

            # add the loss to the running loss
            running_loss+=loss.item()
            
            print('\rEpoch: {}\tbatch: {}\tLoss =  {:.3f}'.format(epoch+1, i, loss), end="")

        running_loss = running_loss / len(train_dataloader)
        report_epoch['training_loss'] = running_loss
        print("\n")
        # validation
        model.eval()
        with torch.no_grad():

            outputs, targets, val_loss = validation(validation_loader=val_dataloader, model= model)
            
            outputs = get_class(outputs)
            

            report_epoch['valid_loss'] = val_loss
            report_epoch['valid_accuracy'] = accuracy_score(targets, outputs)
            report_epoch['valid_f1_macro'] = f1_score(targets, outputs, average='binary')
            print(f"Epoch {epoch+1}: train BCE loss = {running_loss}", 
                  f"|| Valid: BCE loss = {val_loss}   acc = {report_epoch['valid_accuracy']}   macro-F1 = {report_epoch['valid_f1_macro']}")
            

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
        
        # save model each epoch
        torch.save(model.state_dict(), 'epoch'+str(epoch+1)+'.pt')

        print("\n")
        # add performances of the epoch to the overall summary
        summary.append(report_epoch)

    torch.save(dict_model,'Fine_Tuned_Bert_binary_classification_epoch_'+str(epoch+1)+'.pt')
    
    return summary

if __name__ == '__main__':


    train_data, val_data, test_data = tokenize_BERT()


    # batch size is 4 
    train_loader = data_loader(train_data, 4)
    valid_loader = data_loader(val_data, 4)

    # summary get all info about performance
    summary = training_model(nb_epochs = 10, train_dataloader = train_loader, val_dataloader = valid_loader, patience = 5)
    with open('summary.pickle', 'wb') as handle:
        pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)