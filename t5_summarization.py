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
DO_TRAIN = True
NUM_WORKERS = 8

MODEL_PATH = "checkpoints/best-checkpoint-v3.ckpt"

class NewsSummaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        text_max_token_len: int = 1024,
        summary_max_token_len: int = 256
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
    
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

        summary_encoding = tokenizer(
            data_row['summary'],
            max_length=self.summary_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = summary_encoding['input_ids']
        labels[labels == 0] = -100 # to make sure we have correct labels for T5 text generation

        return dict(
            text=text,
            summary=data_row['summary'],
            text_input_ids=text_encoding['input_ids'].flatten(),
            text_attention_mask=text_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding['attention_mask'].flatten()
        )

class NewsSummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = BATCH_SIZE,
        text_max_token_len: int = 1024,
        summary_max_token_len: int = 256
    ):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage=None):
        self.train_dataset = NewsSummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        self.val_dataset = NewsSummaryDataset(
            self.val_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        self.test_dataset = NewsSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
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

class NewsSummaryModel(pl.LightningModule):
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


def summarizeText(text, model):
    text_encoding = tokenizer(
        text,
        max_length=512,
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



tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

data_module = NewsSummaryDataModule(train_df, val_df, test_df, tokenizer)

if DO_TRAIN:
    steps_per_epoch=len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 5

    model = NewsSummaryModel(n_warmup_steps=warmup_steps, n_training_steps=total_training_steps)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints-efever',
        filename='best-checkpoint-t5-summary',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    progress_bar_callback = TQDMProgressBar(refresh_rate=30)
    
    logger = WandbLogger(project='efever-t5', name='t5-base|batch=8|summary')

    trainer = pl.Trainer(
        logger=logger,
        #checkpoint_callback=checkpoint_callback,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
        #progress_bar_refresh_rate=30
    )

    trainer.fit(model, data_module)

    model = NewsSummaryModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

else:
    model = NewsSummaryModel.load_from_checkpoint(MODEL_PATH)

model.freeze()
model.eval()


model_out = []
reference = []
for i in tqdm(range(test_df.shape[0])):
    row = test_df.iloc[i]
    text = str(row['claim']) + str(row['text'])
    model_out.append(summarizeText(text, model))
    reference.append(str(row['summary']))

"""
model_out = ["he began by starting a five person war cabinet and included chamberlain as lord president of the council",
             "the siege lasted from 250 to 241 bc, the romans laid siege to lilybaeum",
             "the original ocean water was found in aquaculture"]

reference = ["he began his premiership by forming a five-man war cabinet which included chamberlain as lord president of the council",
             "the siege of lilybaeum lasted from 250 to 241 bc, as the roman army laid siege to the carthaginian-held sicilian city of lilybaeum",
            "the original mission was for research into the uses of deep ocean water in ocean thermal energy conversion (otec) renewable energy production and in aquaculture"]
"""
rouge = Rouge()

print(rouge.get_scores(model_out, reference, avg=True))
