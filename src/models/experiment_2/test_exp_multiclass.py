import pandas as pd
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import transformers
from sklearn.metrics import f1_score, accuracy_score, classification_report,precision_score, recall_score
import numpy as np
from tokenize_BERT import tokenize_BERT
import pickle


class BERTForFineTuningtWithPooling(torch.nn.Module):
    def __init__(self):
        super(BERTForFineTuningtWithPooling, self).__init__()
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

            ids = data[0].to(device, dtype = torch.long)
            attention_mask = data[1].to(device, dtype = torch.long)
            labels = data[2].to(device, dtype = torch.float)

            output, embedding = model.forward(ids, attention_mask)
            _, predicted = torch.max(output, 1)  
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
        f1_scores[i] = dict_report[str(i)+'.0']['f1-score']
    
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

    precisions = [0] * n_classes
    for i in range(n_classes):
        precisions[i] = dict_report[str(i)+'.0']['precision']

    avg_precision = precision_score(true_labels, predicted_labels, average='macro')

    recalls = [0] * n_classes
    for i in range(n_classes):
        recalls[i] = dict_report[str(i)+'.0']['recall']
    
    avg_recall = recall_score(true_labels, predicted_labels, average='macro')
        
    return counts, class_acc, f1_scores, average_class_acc, overall_acc, macro_f1, precisions, avg_precision, recalls, avg_recall


def testing_pipeline(model_path, testloader, path_to_save_metrics, path_to_save_avg_metrics,  path_to_save_embeddings):
    # test the results
    outputs, targets, embeddings = test(model_path, testloader)

    counts, class_acc, f1_scores, average_class_acc, overall_acc, macro_f1, precisions, avg_precision, recalls, avg_recall = calculate_metrics(outputs, targets)

    avg_info = {'num_sample': targets.shape[0],
                'avg_class_acc': average_class_acc,
                'overall_acc': overall_acc,
                "macro_F1": macro_f1,
                'macro_precison':avg_precision,
                'macro_recall': avg_recall
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

    with open(path_to_save_embeddings+'.pickle', 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__=="__main__":

    # load the test data
    _, _, test_data = tokenize_BERT()

    indexes_test_non_zero_labels = np.where(test_data[3] != 0)[0]

    input_ids_test = test_data[1][indexes_test_non_zero_labels]
    attention_mask_test = test_data[2][indexes_test_non_zero_labels]
    labels_test = test_data[3][indexes_test_non_zero_labels] - 1



    test_dataset = TensorDataset(input_ids_test, attention_mask_test, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    model_path = '/content/drive/MyDrive/epoch1.pt'
    path_to_save_metrics = '/content/drive/MyDrive/metrics_FineTuning_multiclass_targets.csv'
    path_to_save_avg_metrics = '/content/drive/MyDrive/avg_metrics_FineTuning_multiclass_targets.csv'
    path_to_save_embeddings = '/content/drive/MyDrive/embeddings_FineTuning_multiclass_targets'
    
    testing_pipeline(model_path=model_path, testloader=test_loader, path_to_save_metrics=path_to_save_metrics, path_to_save_avg_metrics=path_to_save_avg_metrics, path_to_save_embeddings=path_to_save_embeddings)