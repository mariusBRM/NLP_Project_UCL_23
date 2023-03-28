import pandas as pd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
from tokenize_B import tokenize_BERT
import pickle as pkl

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
    
class BERTForFineTuningtWithPooling(torch.nn.Module):
    def __init__(self, is_pooled):
        super(BERTForFineTuningtWithPooling, self).__init__()
        # first layer is the bert
        self.is_pooled = is_pooled
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        # apply a dropout
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 8)
    
    def forward(self, ids, mask):
        outputs = self.l1(ids, attention_mask=mask)
        if self.is_pooled == True:
            pooled_output = outputs[1]
        else:
            pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        output_2 = self.l2(pooled_output)
        output = self.l3(output_2)

        return pooled_output, output
    
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

# Dataloader
def data_loader(data,batch_size):
    
    # preprocessing
    data = preprocessing(data)

    # Map style for Dataloader
    dataset = HateSpeechData(data)

    # dataloader
    dataloader_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader_loader

def get_class(output):
    l = []
    for pred in output:
        class_pred = [0] * 8
        idx = np.argmax(pred)
        class_pred[idx] = 1.0
        l.append(class_pred)
    return l

def test(model_path, testloader):

    # set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # list containing all the targets and outputs
    targets=[]
    outputs=[]
    embeddings = []
    # load the model need to put either True or false !! 
    model = BERTForFineTuningtWithPooling(is_pooled=True)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # test model
    model.eval()
    with torch.no_grad():
        for data in testloader:

            ids = data['input_ids'].to(device, dtype = torch.long)
            attention_mask = data['attention_mask'].to(device, dtype = torch.long)
            labels = data['labels'].to(device, dtype = torch.float)

            embedding, output = model.forward(ids, attention_mask)
            # adding to list
            targets.extend(labels.cpu().detach().numpy().tolist())
            outputs.extend(output.cpu().detach().numpy().tolist())
            embeddings.extend(embedding.cpu().detach().numpy().tolist())
    
    # get the prediction
    outputs = get_class(outputs)
    outputs = np.array(outputs)
    targets = np.array(targets)
    embeddings = np.array(embeddings)
    
    return outputs, targets, embeddings

def count_elements_per_class(labels):
    num_classes = labels.shape[1]
    class_counts = [0] * num_classes
    
    for i in range(num_classes):
        class_counts[i] = sum(labels[:, i])
        
    return class_counts

def calculate_accuracy(predicted_labels, true_labels):
    num_classes = true_labels.shape[1]
    num_samples = true_labels.shape[0]
    class_counts = count_elements_per_class(true_labels)
    class_accuracies = [0] * num_classes
    true_positives = 0
    
    # calculate accuracy for each class
    for i in range(num_classes):
        true_positives += sum((predicted_labels[:, i] == 1) & (true_labels[:, i] == 1))
        class_accuracies[i] = sum((predicted_labels[:, i] == 1) & (true_labels[:, i] == 1)) / class_counts[i]
    
    # calculate the overall accuracy
    overall_accuracy = true_positives / num_samples
        
    return class_accuracies, overall_accuracy

def calculate_macro_f1(predicted_labels, true_labels):
    num_classes = true_labels.shape[1]
    f1_scores = []
    
    # calculate F1 score for each class
    for i in range(num_classes):
        f1 = f1_score(true_labels[:, i], predicted_labels[:, i])
        f1_scores.append(f1)
        
    # calculate macro F1 score
    macro_f1 = sum(f1_scores) / num_classes
    
    return f1_scores, macro_f1

def report(outputs, targets):
    # calculate the accuracies
    class_accuracies, overall_accuracy = calculate_accuracy(outputs, targets)

    # calculate the f1_scores
    f1_scores, macro_f1 = calculate_macro_f1(outputs, targets)

    # counting the number of sample per class
    sample_per_class = count_elements_per_class(targets)

    report = []

    for i in range(targets.shape[1]):
        info = {
            'num_sample': sample_per_class[i],
            'accuracy': class_accuracies[i],
            'f1_score': f1_scores[i]
        }
        report.append(info)
    
    overall_info = {
        'num_sample': targets.shape[0],
        'accuracy' : overall_accuracy, 
        'f1_score' : macro_f1
    }
    report.append(overall_info)

    return report

def testing_pipeline(model_path, testloader, path_to_save_metrics):
    # test the results
    outputs, targets, embeddings = test(model_path, testloader)

    # calculate the info
    info = report(outputs, targets)

    # save to csv
    df = pd.DataFrame(info)
    df.to_csv(path_to_save_metrics)

    with open('embeddings.pickle', 'wb') as handle:
        pkl.dump(embeddings, handle, protocol=pkl.HIGHEST_PROTOCOL)




if __name__=="__main__":

    model_path = 'model/Fine_Tuned_Bert.pt'
    path_to_save = 'metricsFineTuning.csv'
    # load the datasets
    train_data, val_data, test_data = tokenize_BERT()

    # load the testloader
    testloader = data_loader(test_data, 4)

    # 
    testing_pipeline(model_path=model_path, testloader=testloader, path_to_save_metrics=path_to_save)

