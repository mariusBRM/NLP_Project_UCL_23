from transformers import BertModel
import torch
import pickle
from tokenize_BERT import id_comments, input_ids, attention_masks, labels, bert_model_name, path_dir

# Load model
model = BertModel.from_pretrained(bert_model_name, output_hidden_states = True)
# put the model in evaluation mode
model.eval()

with torch.no_grad():

    outputs = model(input_ids = input_ids, attention_mask = attention_masks)
    hidden_states = outputs[2]

# hidden states shape [layers, batches, tokens, hidden units]
# Number of layers: 13   (initial embeddings + 12 BERT layers)
# Number of batches: number of comments
# Number of tokens: max length of tokenized comments
# Number of hidden units: 768

# create sentence embeddings
# simple approach: average the second to last hidden layer of each token producing a 768 length vectors

# shape [batches, tokens, 768]
token_vecs = hidden_states[-2]

# shape [batches, 768]
sentence_embeddings = torch.mean(token_vecs, dim=1)

# save data to pickle file
data = [id_comments, sentence_embeddings, labels]
with open(path_dir + "/data/hs_sentence_embeddings_no_fine_tuning", "wb") as f:
    pickle.dump(data, f)