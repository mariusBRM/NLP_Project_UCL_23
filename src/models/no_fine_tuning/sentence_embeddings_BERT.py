from transformers import BertModel
import torch
import pickle
from tokenize_BERT import id_comments, input_ids, attention_masks, labels, bert_model_name
from csv import writer


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
    # Number of batches: number of comments
    # Number of tokens: max length of tokenized comments
    # Number of hidden units: 768   

    # shape [batches, tokens, 768]
    token_vecs = hidden_states[-2]

    # shape [batches, 768]
    sentence_embeddings = torch.mean(token_vecs, dim=1)
    
    return sentence_embeddings


if __name__ == '__main__':

    
    # Load model
    model = BertModel.from_pretrained(bert_model_name, output_hidden_states = True)

    # put the model in evaluation mode
    model.eval()


    dico = dict()

    nb_data = input_ids.shape[0]
    batch_size = 200

    nb_full_elements_in_dico = nb_data // batch_size  #nb elements in the dico that have 200 embeddings
    

    with torch.no_grad():
        for i in range(0,nb_full_elements_in_dico):
        
            #First, we are gonna run the bert model nb_elements_in_dico times. Each time, the model takes batch_size embeddings as inputs
        
            print(i)
            sentence_embeddings = get_sentence_embeddings(model, input_ids[i*batch_size:(i+1)*batch_size],attention_masks[i*batch_size:(i+1)*batch_size])
            dico[i+1] = sentence_embeddings

        #then, if there are some remaining embeddings, the model will take remaining_elements embeddings as inputs.
        remaining_embeddings = nb_full_elements_in_dico % batch_size

        if remaining_embeddings > 0:
            #We take care of the remaining embeddings. 
            sentence_embeddings = get_sentence_embeddings(model,input_ids[(i+1)*batch_size:],attention_masks[(i+1)*batch_size:])
        dico[i+2] = sentence_embeddings

    #shape of dico values : (batch_size,768) for the first nb_elements_in_dico elements and (remaining_embeddings,768) for the last element




    #copy results in csv file
    with open('../../../data/hs_sentence_embeddings_no_fine_tuning.csv', 'a') as f_object:
        
        #wite title
        L = ["id_comments", "embedding", "labels"]
        writer_object = writer(f_object)
        writer_object.writerow(L)



        for i in range(0,nb_full_elements_in_dico):
            #print(i)
        
            list_ids = id_comments[i*batch_size : (i+1)*batch_size].to_list() #all id_comments for a batch
            list_labels = labels[i*batch_size : (i+1)*batch_size] #all labels for a batch 
            list_embedd = dico[i+1] #all embeddings for a batch

            for j in range(batch_size):

                L = [list_ids[j],list_embedd[j],list_labels[j]]
                writer_object = writer(f_object)
                writer_object.writerow(L)
    
        if remaining_embeddings > 0 : 

            list_ids = id_comments[(i+1)*200 : ].to_list() #all remaining id_comments
            list_labels = labels[(i+1)*200 :] #all remaining labels 
            list_embedd = dico[i+2] #all embeddings for a batch

            for j in range(len(list_ids)): 

                L = [list_ids[j],list_embedd[j],list_labels[j]]
                writer_object = writer(f_object)
                writer_object.writerow(L)

        
        # Close the file object
        f_object.close()














