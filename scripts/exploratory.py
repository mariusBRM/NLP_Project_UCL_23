# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torchmetrics
from sklearn.model_selection import train_test_split

# %%
import transformers

# %%
dataset_hugging_face = load_dataset("ucberkeley-dlab/measuring-hate-speech", 'binary')
df_train = dataset_hugging_face['train'].to_pandas()
def data_from_hugging_face(df):
    feature = [
        'hate_speech_score',
        'text',
        'target_race',
        'target_religion',
        'target_gender'
    ]
    df = df[feature]
    new_df = df[(df['target_race'] == True) | (df['target_religion'] == True) | (df['target_gender'] == True)]

    return new_df
df = data_from_hugging_face(df_train)



# %%
df.to_pickle('df_hate.pkl')

# %%
train_df, test_df = train_test_split(df, test_size=0.1)


# %%
Classes = ['hate_speech_score', 'target_race', 'target_religion', 'target_gender']

# %%
BERT_MODEL_NAME = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# %%
class ToxicCommentsDataset:#(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        single_row = self.data.iloc[index]
        
        comment = single_row['text']
        labels = single_row[Classes]
        labels[['target_race', 'target_religion', 'target_gender']] = labels[['target_race', 'target_religion', 'target_gender']].astype(int)

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "comment_text": comment,
            "input_ids": encoding["input_ids"].flatten(), # [1,512] => [512]
            "attention_mask": encoding["attention_mask"].flatten(), # [1,512] => [512]
            "labels": torch.FloatTensor(labels)
        }


# %%
class ToxicCommentDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, Classes, batch_size=8, max_len=128):
        super().__init__()
        
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
    
    def setup(self, stage=None):
        self.train_dataset = ToxicCommentsDataset(self.train_df, self.tokenizer, self.max_len)
        self.test_dataset = ToxicCommentsDataset(self.test_df, self.tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=1,num_workers=4)    
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=1,num_workers=4)

# %%
class ToxicCommentClassifier(pl.LightningModule):
    def __init__(self, n_classes: int, steps_per_epoch=None, n_epochs=None):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {
            "loss": loss,
            "predictions": outputs,
            "labels": labels
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)

            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        print("#####")
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(Classes):
            roc_score = torchmetrics.AUROC(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", roc_score, self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]

# %%

# %%



