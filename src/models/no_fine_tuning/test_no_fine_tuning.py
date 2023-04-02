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
from model import Net, preprocessing


    

def test(model_path, testloader):

    # set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # list containing all the targets and outputs
    targets=[]
    outputs=[]

    # load the model
    model = Net(768,8,apply_dropout=False)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # test model
    model.eval()
    with torch.no_grad():
        for data in testloader:

            inputs, labels = data
            

            inputs = inputs.to(device)
            labels = labels.to(device)

            #targets.append(labels)
            
            output = model.forward(inputs)
            
            # adding to list
            _,prediction = torch.max(output, 1)
            #outputs.append(prediction)
            targets.extend(labels.cpu().detach().numpy().tolist())
            outputs.extend(prediction.cpu().detach().numpy().tolist())

            
    
    # get the prediction
    targets = torch.tensor(targets)
    outputs = torch.tensor(outputs)
    #targets = torch.cat(targets, dim=0)
    #outputs = torch.cat(outputs, dim=0)
    
    
    return outputs, targets


def calculate_metrics(predicted_labels, true_labels):


    _, counts = np.unique(true_labels, return_counts=True)

    n_classes = len(counts)
    class_acc = [0] * n_classes
    for i in range(n_classes):
        
        tp = torch.sum((predicted_labels==i) & (true_labels==i))
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


def testing_pipeline(model_path, testloader, path_to_save_metrics, path_to_save_avg_metrics):
    # test the results
    outputs, targets = test(model_path, testloader)

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
    
    overall_info = {
        
        'accuracy' : overall_acc, 
        'f1_score' : macro_f1,
        }
    report.append(overall_info)

    # save to csv
    df = pd.DataFrame(report)
    df.to_csv(path_to_save_metrics)




if __name__=="__main__":

    # load the test data
    df_test = pd.read_csv("../../../data/sentence_embeddings_no_fine_tuning_test.csv")
    embeddings_test = df_test["embedding"].apply(lambda x : preprocessing(x))
    labels_test = df_test["labels"].apply(lambda x : preprocessing(x))

    embeddings_test = torch.stack(embeddings_test.to_list())
    labels_test = torch.stack(labels_test.to_list())

    batch_size = 40

    test_dataset = TensorDataset(embeddings_test, labels_test)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    model_path = './classification_no_fine_tuning.pt'
    path_to_save_metrics = './metrics_no_fineTuning.csv'
    path_to_save_avg_metrics = './avg_metrics_no_fineTuning.csv'
    
    
    testing_pipeline(model_path=model_path, testloader=test_dataloader, path_to_save_metrics=path_to_save_metrics, path_to_save_avg_metrics=path_to_save_avg_metrics)