import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score



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
    df_avg_info.to_csv('avg_metrics.csv')

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