from transformers import BertModel
import torch
import pickle
from tokenize_BERT import train_data, val_data, test_data, bert_model_name
from csv import writer
from torch.utils.data import TensorDataset, DataLoader


if torch.backends.mps.is_available():
    device = mps_device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def get_sentence_embeddings(model,list_input_ids,list_attention_mask):
    """
    This function runs a bert model.
    list_input_ids and list_attention_mask are given as inputs in the model. 

    The function returns a matrix of size [batches, 768], which consists of one embedding of size 768 for each comment.
    """
    
    outputs = model(input_ids = list_input_ids, attention_mask = list_attention_mask)
    hidden_states = outputs[2]
    # hidden states shape [layers, batches, tokens, hidden units]
    # Number of layers: 13   (initial embeddings + 12 BERT layers)
    # batches: size of batch = number of comments
    # Number of tokens: max length of tokenized comments
    # Number of hidden units: 768   

    # take second to last hidden layer
    # shape [batches, tokens, 768]
    token_vecs = hidden_states[-2]

    # average on tokens
    # shape [batches, 768]
    sentence_embeddings = torch.mean(token_vecs, dim=1)
    
    return sentence_embeddings



def store_embeddings_in_a_file(path_to_file, data, model):
    """
    path_to_file : path to the csv file where the embbedings will be stored
    data : (id_comments, inputs_id, attention_mask, labels)
    """

    # put the model in evaluation mode
    model.eval()

    (id_comments, inputs_id, attention_masks, labels) = data
    attention_masks = attention_masks.to(device)

    batch_size = 200
    dataset = TensorDataset(inputs_id, labels)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=False)

    

    with open(path_to_file, 'a') as f_object:

        #write title
        L = ["id_comments", "embedding", "labels"]
        writer_object = writer(f_object)
        writer_object.writerow(L)


        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                mini_batch_inputs_id, mini_batch_labels = data
                mini_batch_inputs_id = mini_batch_inputs_id.to(device)
                mini_batch_labels = mini_batch_labels.to(device)
                
                sentence_embeddings = get_sentence_embeddings(model, mini_batch_inputs_id,attention_masks[i*batch_size:(i+1)*batch_size])


                size_batch = mini_batch_inputs_id.shape[0]

                list_ids = id_comments[i*size_batch : (i+1)*size_batch].to_list() #all id_comments for a batch
                

                for j in range(size_batch):

                    L = [list_ids[j],sentence_embeddings[j],mini_batch_labels[j]]
                    writer_object = writer(f_object)
                    writer_object.writerow(L)
                
                
                
    # Close the file object
    f_object.close()


if __name__ == '__main__':

    model = BertModel.from_pretrained(bert_model_name, output_hidden_states = True)
    model.to(device)

    store_embeddings_in_a_file('../../../data/sentence_embeddings_no_fine_tuning_train__.csv', train_data, model)
    store_embeddings_in_a_file('../../../data/sentence_embeddings_no_fine_tuning_val.csv', val_data, model)
    store_embeddings_in_a_file('../../../data/sentence_embeddings_no_fine_tuning_test.csv', test_data, model)














