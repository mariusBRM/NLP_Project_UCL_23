import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import transformers
from sklearn.metrics import accuracy_score,f1_score
import pickle
from tokenize_BERT import tokenize_BERT


class BERTForFineTuningtWithPooling(torch.nn.Module):
    def __init__(self):
        super(BERTForFineTuningtWithPooling, self).__init__()
        # first layer is the bert
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased', output_hidden_states = False)
        # apply a dropout
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(768, 8)
    
    def forward(self, ids, mask):
        outputs = self.bert(input_ids=ids, attention_mask=mask)
        pooler_output = outputs[1] # torch.mean(outputs.last_hidden_state, dim=1)
        out = self.dropout(pooler_output)
        out = self.linear(out)
        return out

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss(weight=WEIGHTS)(outputs, targets)


def validation(validation_loader, model):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    fin_targets=[]
    fin_outputs=[]
    running_loss = 0.0

    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):

            ids = data[0].to(device, dtype = torch.long)
            mask = data[1].to(device, dtype = torch.long)
            targets = data[2].to(device, dtype = torch.long)
            
            # forward
            output = model.forward(ids, mask)
            # evaluate the loss
            loss = loss_fn(output, targets)


            _, predicted = torch.max(output, 1)  
            # adding to list
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(predicted.cpu().detach().numpy().tolist())

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

            ids = data[0].to(device, dtype = torch.long)
            attention_mask = data[1].to(device, dtype = torch.long)
            labels = data[2].to(device, dtype = torch.long)
            
             # initialize the optimizer
            optimizer.zero_grad()
            #forward inputs
            output = model.forward(ids, attention_mask) 
            # define the loss
            loss = loss_fn(output, labels)
            # backpropagate
            loss.backward()
            # print("Capturing:", torch.cuda.is_current_stream_capturing())
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

            report_epoch['valid_loss'] = val_loss
            report_epoch['valid_accuracy'] = accuracy_score(targets, outputs)
            report_epoch['valid_f1_macro'] = f1_score(targets, outputs, average='macro')
            print(f"Epoch {epoch+1}: train CE loss = {running_loss}", 
                  f"|| Valid: CE loss = {val_loss}   acc = {report_epoch['valid_accuracy']}   macro-F1 = {report_epoch['valid_f1_macro']}")
            

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

    torch.save(dict_model, 'Fine_Tuned_Bert.pt')
    
    return summary



if __name__ == '__main__':

    train_data, val_data, _ = tokenize_BERT()
    WEIGHTS = 1 / (torch.sqrt(torch.unique(train_data[3], return_counts = True)[1])).to('cuda')


    # create datasets
    train_dataset = TensorDataset(train_data[1],train_data[2], train_data[3])
    val_dataset = TensorDataset(val_data[1], val_data[2], val_data[3])
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # summary get all info about performance
    summary = training_model(nb_epochs = 10, train_dataloader = train_loader, val_dataloader = val_loader, patience = 5)
    with open('summary.pickle', 'wb') as handle:
        pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)
