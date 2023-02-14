import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import pandas as pd
import shutil
import sys
import os
import dataloader
import preprocessing 
import sklearn.model_selection as sklm

current_directory = os.getcwd()
os.chdir("..")
file_models = os.getcwd() + '\models'
file_checkpoints = os.getcwd() + '\checkspoints'


def load_data():
    # load datset
    dataset = dataloader.loader_data()
    df_train = dataset['train'].to_pandas()
    df = preprocessing.data_from_hugging_face(df_train)
    # defining y & X
    X = df['text']
    y = df[['hate_speech_score', 'target_race', 'target_religion', 'target_gender']]
    y[['target_race', 'target_religion', 'target_gender']] = y[['target_race', 'target_religion', 'target_gender']].astype(int)
    # split into train/validation/test
    X_train, X_test, y_train, y_test  = sklm.train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = sklm.train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    return X_train, y_train, X_test, y_test, X_val, y_val

# Tokenizer 
class Tokenizer():
    def __init__(self, X, y, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.X = X
        self.y = y.values
        self.max_len = max_len
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        print(self.X[index])
        text = str(self.X[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.y[index])
        }



# save our model's checkpoint
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


# load the model's checkpoint
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 4)
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

# train our model
def train_model(n_epochs, training_loader, validation_loader, model, 
                optimizer, checkpoint_path, best_model_path, device, val_targets, val_outputs):
   
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    # looping over the 
    for epoch in range(1, n_epochs+1):
        train_loss = 0
        valid_loss = 0

        model.train()
        print('############# Epoch {}: Training Start   #############'.format(epoch))
        for batch_idx, data in enumerate(training_loader):
            #print('yyy epoch', batch_idx)
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            #if batch_idx%5000==0:
            #   print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('before loss data in training', loss.item(), train_loss)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            #print('after loss data in training', loss.item(), train_loss)
        
        print('############# Epoch {}: Training End     #############'.format(epoch))
        
        print('############# Epoch {}: Validation Start   #############'.format(epoch))
    ######################    
    # validate the model #
    ######################
    model.eval()
   
    with torch.no_grad():
      for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

      print('############# Epoch {}: Validation End     #############'.format(epoch))
      # calculate average losses
      #print('before cal avg train loss', train_loss)
      train_loss = train_loss/len(training_loader)
      valid_loss = valid_loss/len(validation_loader)
      # print training/validation statistics 
      print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
      
      # create checkpoint variable and add important data
      checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
      }
        
        # save checkpoint
      save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
      ## TODO: save the model if validation loss has decreased
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss

    print('############# Epoch {}  Done   #############\n'.format(epoch))

    return model


def main():

    # hyperparameters
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    EPOCHS = 2
    LEARNING_RATE = 1e-05
    # load data
    X_train, y_train, X_test, y_test, X_val, y_val = load_data()
    
    # tokenizer from pretrained Bert
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize our dataset
    train_dataset = Tokenizer(X_train, y_train, tokenizer, MAX_LEN)
    valid_dataset = Tokenizer(X_val, y_val, tokenizer, MAX_LEN)

    # provide the iterable dataset over the training batch & validation batch
    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_data_loader = torch.utils.data.DataLoader(valid_dataset, 
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    # Set GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # initiate the model
    model = BERTClass()
    model.to(device)
    # define the optimizer
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
    # 
    val_targets=[]
    val_outputs=[]  
    
    return train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, file_models, file_checkpoints, device, val_targets, val_outputs)
    


if __name__=="__main__":
    sys.exit(main())