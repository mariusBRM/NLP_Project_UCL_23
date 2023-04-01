import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import pickle
from torch.utils.data import TensorDataset, DataLoader
from tokenize_BERT import tokenize_BERT
from model_binary_classification import get_class, data_loader
from prepare_data import *


class BERTForFineTuningtWithPooling_Binary(torch.nn.Module):
    def __init__(self):
        super(BERTForFineTuningtWithPooling_Binary, self).__init__()
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

def test_binary(model_path, testloader):

    # set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # list containing all the targets and outputs
    targets = []
    outputs = []
    embeddings = []
    comment_id = []

    model = BERTForFineTuningtWithPooling_Binary()
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
            comment_id.extend(data['comment_id'].numpy().tolist())
    
    # get the prediction
    outputs = np.array(outputs)
    targets = np.array(targets)
    embeddings = np.array(embeddings)
    comment_id =np.array(comment_id)
    
    return outputs, targets, embeddings, comment_id

def get_pred_correct_binary(outputs, targets, comment_id):

    idx_pred_hateful = np.squeeze(np.argwhere(outputs == 1))
    comment_id_hateful = comment_id[idx_pred_hateful]

    tn, fp, fn, tp = confusion_matrix(targets, outputs).ravel()

    return comment_id_hateful, fp/len(outputs), fn/len(outputs)


class BERTForFineTuningtWithPooling_multiclass(torch.nn.Module):
    def __init__(self):
        super(BERTForFineTuningtWithPooling_multiclass, self).__init__()
        # first layer is the bert
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased', output_hidden_states = False)
        # apply a dropout
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(768, 7)
    
    def forward(self, ids, mask):
        outputs = self.bert(input_ids=ids, attention_mask=mask)
        pooler_output = outputs[1] # torch.mean(outputs.last_hidden_state, dim=1)
        out = self.dropout(pooler_output)
        out = self.linear(out)
        return out, pooler_output
    

def test_multiclass(model_path, testloader):

    # set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # list containing all the targets and outputs
    targets = []
    outputs = []
    embeddings = []
    comment_id = []

    model = BERTForFineTuningtWithPooling_multiclass()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # test model
    model.eval()
    with torch.no_grad():
        for data in testloader:

            ids = data[0].to(device, dtype = torch.long)
            attention_mask = data[1].to(device, dtype = torch.long)
            labels = data[2].to(device, dtype = torch.float)

            output, embedding = model.forward(ids, attention_mask)
            _, predicted = torch.max(output, 1)  
            # adding to list
            targets.extend(labels.cpu().detach().numpy().tolist())
            outputs.extend(predicted.cpu().detach().numpy().tolist())
            embeddings.extend(embedding.cpu().detach().numpy().tolist())
            comment_id.extend(data[3].numpy().tolist())
    
    # get the prediction
    outputs = np.array(outputs)
    targets = np.array(targets)
    embeddings = np.array(embeddings)
    comment_id = np.array(comment_id)
    
    return outputs, targets, embeddings, comment_id


def calculate_metrics_multiclass(predicted_labels, true_labels):

    _, counts = np.unique(true_labels, return_counts=True)

    n_classes = len(counts)
    class_acc = [0] * n_classes
    for i in range(n_classes):

        tp = np.sum((predicted_labels==i) & (true_labels==i))
        class_acc[i] = tp / counts[i]

    average_class_acc = np.mean(class_acc)
    overall_acc = accuracy_score(true_labels, predicted_labels)
    
    f1_scores = [0] * n_classes
    dict_report = classification_report(true_labels, predicted_labels, output_dict=True)
    for i in range(n_classes):
        f1_scores[i] = dict_report[str(i)]['f1-score']
    
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

    precisions = [0] * n_classes
    for i in range(n_classes):
        precisions[i] = dict_report[str(i)]['precision']

    avg_precision = precision_score(true_labels, predicted_labels, average='macro')

    recalls = [0] * n_classes
    for i in range(n_classes):
        recalls[i] = dict_report[str(i)]['recall']
    
    avg_recall = recall_score(true_labels, predicted_labels, average='macro')
        
    return counts, class_acc, f1_scores, average_class_acc, overall_acc, macro_f1, precisions, avg_precision, recalls, avg_recall


def testing_pipeline(outputs, targets, path_to_save_metrics, path_to_save_avg_metrics, fpr, fnr):
    # test the results

    counts, class_acc, f1_scores, average_class_acc, overall_acc, macro_f1, precisions, avg_precision, recalls, avg_recall = calculate_metrics_multiclass(outputs, targets)

    avg_info = {'num_sample': targets.shape[0],
                'avg_class_acc': average_class_acc,
                'overall_acc': overall_acc,
                "macro_F1": macro_f1,
                'macro_precison':avg_precision,
                'macro_recall': avg_recall,
                'binary model FPR': fpr,
                'binary model FNR': fnr
                }
    # save to csv
    df_avg_info = pd.DataFrame(avg_info, index=[0])
    df_avg_info.to_csv(path_to_save_avg_metrics)


    report = []

    for i in range(len(counts)):
        info = {
            'num_sample': counts[i],
            'accuracy': class_acc[i],
            'f1_score': f1_scores[i], 
            'precision' : precisions[i],
            'recall':recalls[i]
            }
        report.append(info)
    
    

    # save to csv
    df = pd.DataFrame(report)
    df.to_csv(path_to_save_metrics)


if __name__ == '__main__':


    _, _, test_data = tokenize_BERT()


    # batch size is 4 
    test_loader_binary = data_loader(test_data, 4)

    # create dicitonary of labels
    dict_label = {}
    for i in range(len(test_data[0])):
        dict_label[test_data[0][i]] = int(test_data[3][i])

    model_path_binary = 'Fine_Tuned_Bert_binary.pt'
    model_path_multiclass = 'multiclass_targets.pt'

    outputs, targets, _, comment_id = test_binary(model_path_binary, test_loader_binary)

    comment_id_hateful, fpr, fnr = get_pred_correct_binary(outputs, targets, comment_id)


    dict_pred = {}

    # get data of positive predictions
    comment_id_multiclass = []
    inputs_id_multiclass = []
    attention_masks_multiclass = []
    labels_multiclass = []
    # loop over all comment ID
    for i in range(len(test_data[0])):
        # if comment is hateful
        if test_data[0][i] in comment_id_hateful:
            comment_id_multiclass.append(test_data[0][i])
            inputs_id_multiclass.append(test_data[1][i])
            attention_masks_multiclass.append(test_data[2][i])
            labels_multiclass.append(test_data[3][i] - 1)
        # if not hateful, add to dictionary of predictions
        else:
            dict_pred[test_data[0][i]] = 0


    test_dataset_multiclass = TensorDataset(torch.stack(inputs_id_multiclass), torch.stack(attention_masks_multiclass), torch.Tensor(labels_multiclass), torch.Tensor(comment_id_multiclass))
    test_loader_multiclass = DataLoader(test_dataset_multiclass, batch_size=4, shuffle=False, num_workers=0)

    outputs, _, _, comment_id_targets = test_multiclass(model_path_multiclass, test_loader_multiclass)

    for i in range(len(comment_id_targets)):
        dict_pred[comment_id_targets[i]] = outputs[i] + 1


    final_out = []
    final_tar = []

    for key in dict_label:
        final_out.append(dict_pred[key])
        final_tar.append(dict_label[key])


    path_to_save_metrics = 'metrics_exp2_both.csv'
    path_to_save_avg_metrics = 'avg_metrics_exp2_both.csv'
    
    testing_pipeline(outputs=np.array(final_out), targets=np.array(final_tar), path_to_save_metrics=path_to_save_metrics, path_to_save_avg_metrics=path_to_save_avg_metrics, fpr=fpr, fnr=fnr)

