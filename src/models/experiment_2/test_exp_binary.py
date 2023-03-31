import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
from tokenize_BERT import tokenize_BERT
from model_binary_classification import get_class, data_loader
from prepare_data import *


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
        return torch.squeeze(output), pooled_output
    

def test(model_path, testloader):

    # set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # list containing all the targets and outputs
    targets = []
    outputs = []
    embeddings = []

    model = BERTForFineTuningtWithPooling()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # test model
    model.eval()
    with torch.no_grad():
        for data in testloader:

            ids = data['input_ids'].to(device, dtype = torch.long)
            attention_mask = data['attention_mask'].to(device, dtype = torch.long)
            labels = data['labels'].to(device, dtype = torch.float)

            output, embedding = model.forward(ids, attention_mask)
            predicted = get_class(output)
            # adding to list
            targets.extend(labels.cpu().detach().numpy().tolist())
            outputs.extend(predicted.cpu().detach().numpy().tolist())
            embeddings.extend(embedding.cpu().detach().numpy().tolist())
    
    # get the prediction
    outputs = np.array(outputs)
    targets = np.array(targets)
    embeddings = np.array(embeddings)
    
    return outputs, targets, embeddings


def calculate_metrics(predicted_labels, true_labels):

    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, pos_label=1, average='binary')
    precision = precision_score(true_labels, predicted_labels, pos_label=1, average='binary')
    recall = recall_score(true_labels, predicted_labels, pos_label=1, average='binary')
        
    return precision, recall, acc, f1


def testing_pipeline(model_path, testloader, path_to_save_avg_metrics,  path_to_save_embeddings):
    # test the results
    outputs, targets, embeddings = test(model_path, testloader)

    precision, recall, acc, f1 = calculate_metrics(outputs, targets)

    avg_info = {'precision': precision,
                'recall': recall,
                "accuracy": acc,
                "F1_score": f1
                }
    # save to csv
    df_avg_info = pd.DataFrame(avg_info, index=[0])
    df_avg_info.to_csv(path_to_save_avg_metrics)

    with open(path_to_save_embeddings+'.pickle', 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == '__main__':


    _, _, test_data = tokenize_BERT()


    # batch size is 4 
    test_loader = data_loader(test_data, 4)

    model_path = 'Fine_Tuned_Bert_binary.pt'
    path_to_save_avg_metrics = 'binary_metrics'
    path_to_save_embeddings = 'binary_embeddings'

    testing_pipeline(model_path, testloader=test_loader, path_to_save_avg_metrics=path_to_save_avg_metrics,  path_to_save_embeddings=path_to_save_embeddings)