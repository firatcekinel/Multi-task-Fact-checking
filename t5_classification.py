import json
import pandas as pd
import numpy as np
import torch
torch.cuda.empty_cache()
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.model_selection import train_test_split
#from termcolor import colored
import datetime
import wandb
import pickle
from rouge import Rouge
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

pl.seed_everything(42)

MODEL_NAME = 't5-base'
N_EPOCHS = 3
BATCH_SIZE = 8
DO_TRAIN = False
NUM_WORKERS = 8
label_dict = {"supports": 0, "refutes": 1, "not enough info": 2}
MODEL_PATH = "checkpoints-efever/best-checkpoint-t5-classification.ckpt"

class NewsClassificationDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        text_max_token_len: int = 1024, 
        label_max_token_len: int = 4
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.label_max_token_len = label_max_token_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row['claim'] + data_row['text']

        text_encoding = tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        label_encoding = tokenizer(
            data_row['label'],
            max_length=self.label_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        return dict(
            text=text,
            label=data_row['label'],
            text_input_ids=text_encoding['input_ids'].flatten(),
            text_attention_mask=text_encoding['attention_mask'].flatten(),
            labels=label_encoding['input_ids'].flatten(),
            labels_attention_mask=label_encoding['attention_mask'].flatten()
        )

class NewsClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = BATCH_SIZE,
        text_max_token_len: int = 1024,
        label_max_token_len: int = 4
    ):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.label_max_token_len = label_max_token_len

    def setup(self, stage=None):
        self.train_dataset = NewsClassificationDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.label_max_token_len
        )
        self.val_dataset = NewsClassificationDataset(
            self.val_df,
            self.tokenizer,
            self.text_max_token_len,
            self.label_max_token_len
        )
        self.test_dataset = NewsClassificationDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.label_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            #shuffle=True,
            num_workers=NUM_WORKERS
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            #shuffle=True,
            num_workers=NUM_WORKERS
        )

class NewsClassificationModel(pl.LightningModule):
    def __init__(self, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
    
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        #return AdamW(self.parameters(), lr=0.0001)

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=self.n_warmup_steps,
          num_training_steps=self.n_training_steps
        )

        return dict(
          optimizer=optimizer,
          lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
          )
        )


def decodeText(text, model):
    text_encoding = tokenizer(
        text,
        max_length=1024,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    generated_ids = model.model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [
            tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
    ]
    return "".join(preds)



train_df = pd.read_csv("../summary/FEVER/efever_train.tsv", sep="\t", encoding='utf-8') #
val_df = pd.read_csv("../summary/FEVER/efever_dev.tsv", sep="\t", encoding='utf-8')
test_df = pd.read_csv("../summary/FEVER/efever_test.tsv", sep="\t", encoding='utf-8')

train_df = train_df[['claim', 'summary', 'retrieved_evidence', 'label']]
train_df = train_df.dropna()
train_df.columns = ['claim', 'summary', 'text', 'label']
train_df = train_df[train_df.label != "snopes"]

val_df = val_df[['claim', 'summary', 'retrieved_evidence', 'label']]
val_df = val_df.dropna()
val_df.columns = ['claim', 'summary', 'text', 'label']
val_df = val_df[val_df.label != "National, Candidate Biography, Donald Trump,"]

test_df = test_df[['claim', 'summary', 'retrieved_evidence', 'label']]
test_df = test_df.dropna()
test_df.columns = ['claim', 'summary', 'text', 'label']


#train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
#test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

data_module = NewsClassificationDataModule(train_df, val_df, test_df, tokenizer)

if DO_TRAIN:
    steps_per_epoch=len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 5

    model = NewsClassificationModel(n_warmup_steps=warmup_steps, n_training_steps=total_training_steps)
    #model = NewsClassificationModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints-efever',
        filename='best-checkpoint-t5-classification',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    progress_bar_callback = TQDMProgressBar(refresh_rate=30)
    logger = WandbLogger(project='efever-t5', name='t5-base|batch=8|classification')

    trainer = pl.Trainer(
        logger=logger,
        #checkpoint_callback=checkpoint_callback,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
        #progress_bar_refresh_rate=30
    )

    trainer.fit(model, data_module)

    model = NewsClassificationModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

else:
    model = NewsClassificationModel.load_from_checkpoint(MODEL_PATH)

model.freeze()
model.eval()


print("CLASSIFICATION RESULTS")
model_out = []
reference = []

for i in tqdm(range(test_df.shape[0])):
    row = test_df.iloc[i]
    text = str(row['claim']) + str(row['text'])

    pred = decodeText(text, model).lower()
    try:
        model_out.append(label_dict[pred])
    except:
        for k in label_dict.keys():
            if k.startswith(pred):
                model_out.append(label_dict[k])
    reference.append(label_dict[str(row['label']).lower()])


print(confusion_matrix(reference, model_out))

print("f1-macro: ", f1_score(reference, model_out, average='macro'))
print("f1-micro: ", f1_score(reference, model_out, average='micro'))
print("f1-weighted: ", f1_score(reference, model_out, average='weighted'))
print("prec-score: ", precision_score(reference, model_out, average='weighted'))
print("recall-score: ", recall_score(reference, model_out, average='weighted'))
print("acc-score: ", accuracy_score(reference, model_out))
