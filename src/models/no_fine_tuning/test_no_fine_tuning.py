
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
#import sys
#sys.path.append("../models/no_fine_tuning")
from model import Net, preprocessing
import torch
import numpy as np
import pandas as pd

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
            targets.append(labels)
            
            output = model.forward(inputs)
            
            # adding to list
            _,prediction = torch.max(output, 1)
            outputs.append(prediction)

            
    
    # get the prediction
    targets = torch.cat(targets, dim=0)
    outputs = torch.cat(outputs, dim=0)
    
    identity = np.eye(8)
    outputs = identity[outputs]
    targets = identity[targets]
    
    return outputs, targets






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

    avg_info = {'avg_class_acc': np.mean(class_accuracies),
                'overall_acc': overall_accuracy,
                "macro_F1": macro_f1
                }
    
    # save to csv
    df_avg_info = pd.DataFrame(avg_info, index=[0])
    df_avg_info.to_csv('avg_metrics_no_fine_tuning.csv')

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

def get_results_on_csv(path_to_save_metrics, outputs, targets):
    
    # calculate the info
    info = report(outputs, targets)

    # save to csv
    df = pd.DataFrame(info)
    df.to_csv(path_to_save_metrics)











if __name__ == '__main__': 

    df_test = pd.read_csv("../../../data/sentence_embeddings_no_fine_tuning_test.csv")
    embeddings_test = df_test["embedding"].apply(lambda x : preprocessing(x))
    labels_test = df_test["labels"].apply(lambda x : preprocessing(x))

    embeddings_test = torch.stack(embeddings_test.to_list())
    labels_test = torch.stack(labels_test.to_list())

    batch_size = 40

    test_dataset = TensorDataset(embeddings_test, labels_test)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    outputs, targets = test("./classification_no_fine_tuning.pt", test_dataloader)

    get_results_on_csv("./metrics_no_fine_tuning.csv", outputs, targets)


