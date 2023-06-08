import pandas as pd
import numpy as np
import torch
torch.cuda.empty_cache()
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.model_selection import train_test_split
#from termcolor import colored
from rouge import Rouge
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report, multilabel_confusion_matrix

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

pl.seed_everything(42)

MODEL_NAME = 't5-base'
N_EPOCHS = 5
BATCH_SIZE = 4
DO_TRAIN = True
NUM_WORKERS = 4
SUM_LOSS_COEFF = 0.2
CL_LOSS_COEFF = 0.8
MIXTURE_COEFF = 1
UNPROVEN_COEFF = 1
CL_HIDDEN_SIZE = 32
CL_DROPOUT_PROB = 0.1

TEXT_TOKEN_LEN=1024
SUMMARY_TOKEN_LEN=256
LABEL_TOKEN_LEN=4
LR=1e-4

UNCERTAINITY_LOSS=False

#label_dict = {"unproven": 0, "false": 1, "mixture": 2, "true": 3}
label_dict = {"supports": 0, "refutes": 1, "not enough info": 2}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXP_NAME = "uncertainity_loss=" + str(UNCERTAINITY_LOSS) + "|model=" + MODEL_NAME + "|batch=" + str(BATCH_SIZE) + "|epoch=" + str(N_EPOCHS) + "|sum_coeff=" + str(SUM_LOSS_COEFF) + "|cl_coeff=" + str(CL_LOSS_COEFF) + "|mixture_coeff=" + str(MIXTURE_COEFF) + "|unproven_coeff=" + str(UNPROVEN_COEFF) + "|text_token_len=" + str(TEXT_TOKEN_LEN) + "|lr=" + str(LR) + "|cl_hidden_size=" + str(CL_HIDDEN_SIZE) + "|cl_dropout_prob=" + str(CL_DROPOUT_PROB)

model_path="checkpoints-efever/best-checkpoint-v8.ckpt"

class NewsDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        text_max_token_len: int = TEXT_TOKEN_LEN,
        summary_max_token_len: int = SUMMARY_TOKEN_LEN,
        label_max_token_len: int = LABEL_TOKEN_LEN
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
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

        summary_encoding = tokenizer(
            data_row['summary'],
            max_length=self.summary_max_token_len,
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

        labels = summary_encoding['input_ids']
        labels[labels == 0] = -100 # to make sure we have correct labels for T5 text generation
        
        cl_labels = label_encoding['input_ids']
        cl_labels[cl_labels == 0] = -100

        return dict(
            text=text,
            summary=data_row['summary'],
            label=label_dict[data_row['label'].lower()],
            text_input_ids=text_encoding['input_ids'].flatten(),
            text_attention_mask=text_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding['attention_mask'].flatten(),
            #cl_labels=cl_labels.flatten(),
            #cl_labels_attention_mask=label_encoding['attention_mask'].flatten()
        )

class NewsMTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = BATCH_SIZE,
        text_max_token_len: int = TEXT_TOKEN_LEN,
        summary_max_token_len: int = SUMMARY_TOKEN_LEN,
        label_max_token_len: int = LABEL_TOKEN_LEN
    ):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
        self.label_max_token_len = label_max_token_len

    def setup(self, stage=None):
        self.train_dataset = NewsDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len,
            self.label_max_token_len
        )
        self.val_dataset = NewsDataset(
            self.val_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len,
            self.label_max_token_len
        )
        self.test_dataset = NewsDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len,
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

class NewsMTModel(pl.LightningModule):
    def __init__(self, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        self.classifier = torch.nn.Linear(self.model.config.d_model, CL_HIDDEN_SIZE)
        self.classifier2 = torch.nn.Linear(CL_HIDDEN_SIZE, 4)
        self.dropout = torch.nn.Dropout(CL_DROPOUT_PROB) 
        self.criterion = torch.nn.CrossEntropyLoss()

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        
        self.sigma1 = torch.nn.Parameter(torch.zeros(1))
        self.sigma2 = torch.nn.Parameter(torch.zeros(1))

    
    def forward(self, input_ids, attention_mask, decoder_attention_mask=None, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        cl_output = self.dropout(F.relu(self.classifier(output.encoder_last_hidden_state.mean(dim=1))))
        cl_output = self.classifier2(cl_output)
        cl_output = torch.sigmoid(cl_output)

        return output.loss, output.logits, cl_output, [self.sigma1, self.sigma2]

    def training_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        summary_labels = batch['labels']
        summary_attention_mask = batch['labels_attention_mask']
        #cl_labels = batch['cl_labels']
        #cl_attention_mask = batch['cl_labels_attention_mask']

        summary_loss, summary_logits, cl_output, log_vars = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=summary_attention_mask,
            labels=summary_labels,
        )

        cl_loss = self.criterion(cl_output, batch["label"])
        
        if UNCERTAINITY_LOSS:
            pre1 = torch.exp(-log_vars[0])
            pre2 = torch.exp(-log_vars[1])
            loss = torch.sum(pre1*summary_loss + log_vars[0], -1)
            loss += torch.sum(pre2*cl_loss + log_vars[1], -1)
            loss = torch.mean(loss)
        else:
            loss = SUM_LOSS_COEFF*summary_loss + CL_LOSS_COEFF*cl_loss
		
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_summary_loss", summary_loss, prog_bar=True, logger=True)
        self.log("train_classification_loss", cl_loss, prog_bar=True, logger=True)
        return {"loss": loss,
                "cl_labels": batch["label"], "cl_predictions": cl_output, 
                "target_summaries":summary_labels,   "predicted_summaries":summary_logits,
                }
    
    def validation_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        summary_labels = batch['labels']
        summary_attention_mask = batch['labels_attention_mask']
        #cl_labels = batch['cl_labels']
        #cl_attention_mask = batch['cl_labels_attention_mask']

        summary_loss, summary_logits, cl_output, log_vars = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=summary_attention_mask,
            labels=summary_labels,
        )


        cl_loss = self.criterion(cl_output, batch["label"])
        
        if UNCERTAINITY_LOSS:
            pre1 = torch.exp(-log_vars[0])
            pre2 = torch.exp(-log_vars[1])
            loss = torch.sum(pre1*summary_loss + log_vars[0], -1)
            loss += torch.sum(pre2*cl_loss + log_vars[1], -1)
            loss = torch.mean(loss)
        else:
            loss = SUM_LOSS_COEFF*summary_loss + CL_LOSS_COEFF*cl_loss

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_summary_loss", summary_loss, prog_bar=True, logger=True)
        self.log("val_classification_loss", cl_loss, prog_bar=True, logger=True)
        
        return loss

    def test_step(self, batch, batch_size):
        global model_out
        global reference
        
        _, _, cl_preds, log_vars = self(input_ids=batch["text_input_ids"], 
                            attention_mask=batch["text_attention_mask"], 
                            labels=batch["labels"], 
                            decoder_attention_mask=batch['labels_attention_mask'])

        y_preds = torch.max(cl_preds, axis=1).indices.detach().cpu().tolist()
        y_true = batch["label"].detach().cpu().tolist()

        model_out = model_out + y_preds
        reference = reference + y_true
    
    def configure_optimizers(self):
        #return AdamW(self.parameters(), lr=0.0001)

        optimizer = AdamW(self.parameters(), lr=LR)

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
        max_length=TEXT_TOKEN_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    generated_ids = model.model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=SUMMARY_TOKEN_LEN,
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

train_df = pd.read_csv("../summary/FEVER/efever_train_binary.tsv", sep="\t", encoding='utf-8') #
val_df = pd.read_csv("../summary/FEVER/efever_dev_binary.tsv", sep="\t", encoding='utf-8')
#train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
test_df = pd.read_csv("../summary/FEVER/efever_test_binary.tsv", sep="\t", encoding='utf-8')

train_df = train_df[['claim', 'summary', 'retrieved_evidence', 'label']]
train_df = train_df.dropna()
train_df.columns = ['claim', 'summary', 'text', 'label']
train_df['text'] = train_df['text'].apply(lambda x: x.replace("+", ""))

val_df = val_df[['claim', 'summary', 'retrieved_evidence', 'label']]
val_df = val_df.dropna()
val_df.columns = ['claim', 'summary', 'text', 'label']
val_df['text'] = val_df['text'].apply(lambda x: x.replace("+", ""))

test_df = test_df[['claim', 'summary', 'retrieved_evidence', 'label']]
test_df = test_df.dropna()
test_df.columns = ['claim', 'summary', 'text', 'label']
test_df['text'] = test_df['text'].apply(lambda x: x.replace("+", ""))


#train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
#test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

data_module = NewsMTDataModule(train_df, val_df, test_df, tokenizer)

if DO_TRAIN:
    steps_per_epoch=len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 5

    model = NewsMTModel(n_warmup_steps=warmup_steps, n_training_steps=total_training_steps)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints-efever',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    progress_bar_callback = TQDMProgressBar(refresh_rate=30)

    logger = WandbLogger(project='efever-mt-uncertainity', name=EXP_NAME)

    trainer = pl.Trainer(
        logger=logger,
        #checkpoint_callback=checkpoint_callback,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
        #progress_bar_refresh_rate=30
    )

    trainer.fit(model, data_module)
    
    model_path = trainer.checkpoint_callback.best_model_path
    model = NewsMTModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)


else:
    model = NewsMTModel.load_from_checkpoint(model_path)


model.freeze()
model.eval()


model_out = []
reference = []

for i in tqdm(range(test_df.shape[0])):
    row = test_df.iloc[i]
    text = str(row['claim']) + str(row['text'])
    model_out.append(summarizeText(text, model))
    reference.append(str(row['summary']))

rouge = Rouge()
print("SUMMARY RESULTS")
print(rouge.get_scores(model_out, reference, avg=True))



print("CLASSIFICATION RESULTS")
model_out = []
reference = []
trainer.test(model, data_module)



print(confusion_matrix(reference, model_out))

print("f1-macro: ", f1_score(reference, model_out, average='macro'))
print("f1-micro: ", f1_score(reference, model_out, average='micro'))
print("f1-weighted: ", f1_score(reference, model_out, average='weighted'))
print("prec-score: ", precision_score(reference, model_out, average='weighted'))
print("recall-score: ", recall_score(reference, model_out, average='weighted'))
print("acc-score: ", accuracy_score(reference, model_out))

params = EXP_NAME + "|model_path=" +model_path
print(params)
