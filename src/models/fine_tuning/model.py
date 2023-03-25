import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from sklearn.metrics import accuracy_score,f1_score
from sklearn.utils.class_weight import compute_class_weight
import pickle
from tokenize_BERT import tokenize_BERT

# import + preprocess the data
def preprocessing(tuple):     
    # changing labels 0,...,7 to one-hot encoded list
    labels = tuple[3]
    l = []
    for i in range(len(labels)):
        list_class = [0] * 8
        list_class[int(labels[i])] = 1
        l.append(list_class)
        
    new_tuple = (tuple[0], tuple[1], tuple[2], torch.tensor(l))
    return new_tuple

def get_class(output):
    l = []
    for pred in output:
        class_pred = [0] * 8
        idx = np.argmax(pred)
        class_pred[idx] = 1.0
        l.append(class_pred)
    return l

train_data, val_data, test_data = tokenize_BERT()
WEIGHTS = compute_class_weight(class_weight='balanced', classes=np.unique(train_data[3]), y=train_data[3].numpy())
WEIGHTS = torch.tensor(WEIGHTS, dtype=torch.float).to('cuda')

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
        # create dictionnary
        item = {
            'input_ids':inputs_id,
            'attention_mask':attention_mask,
            'labels':label
        }
        return item
    
    def __len__(self):
        return len(self.X[1])
    

# Dataloader
def data_loader(data,batch_size):
    
    # preprocessing
    data = preprocessing(data)

    # Map style for Dataloader
    dataset = HateSpeechData(data)

    # dataloader
    dataloader_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader_loader

class BERTForFineTuningtWithPooling(torch.nn.Module):
    def __init__(self):
        super(BERTForFineTuningtWithPooling, self).__init__()
        # first layer is the bert
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        # apply a dropout
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 8)
    
    def forward(self, ids, mask):
        outputs = self.l1(ids, attention_mask=mask)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        output_2 = self.l2(pooled_output)
        output = self.l3(output_2)
        return outputs.hidden_states, output

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss(pos_weight=WEIGHTS)(outputs, targets)


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
            _,output = model.forward(ids, mask)
            # evaluate the loss
            loss = loss_fn(output, targets)

            # adding to list
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())

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
            _, output = model.forward(ids, attention_mask)
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
            # getting the predominant class
            outputs = get_class(outputs)
            outputs = np.array(outputs)

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

    # batch size is 4 
    train_loader = data_loader(train_data, 4)
    valid_loader = data_loader(val_data, 4)

    # summary get all info about performance
    summary = training_model(nb_epochs = 20, train_dataloader = train_loader, val_dataloader = valid_loader, patience = 5)
    with open('summary.pickle', 'wb') as handle:
        pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)
