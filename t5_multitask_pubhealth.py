import pandas as pd
import numpy as np
import torch
torch.cuda.empty_cache()
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.model_selection import train_test_split
import wandb
import argparse
from rouge import Rouge
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report, multilabel_confusion_matrix

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

pl.seed_everything(42)

# default params
NUM_WORKERS = 4
CL_DROPOUT_PROB = 0.1
TEXT_TOKEN_LEN=512+256
SUMMARY_TOKEN_LEN=208
LABEL_TOKEN_LEN=4

label_dict = {"unproven": 3, "false": 0, "mixture": 1, "true": 2}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path="checkpoints-pubhealth/best-checkpoint-v3.ckpt"

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
            data_row["label"],
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
            label=label_dict[data_row['label']],
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
        batch_size: int = 1,
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
        self.classifier2 = torch.nn.Linear(CL_HIDDEN_SIZE, LABEL_TOKEN_LEN)
        self.dropout = torch.nn.Dropout(CL_DROPOUT_PROB) 
        self.criterion = torch.nn.CrossEntropyLoss()

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        
    
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

        return output.loss, output.logits, cl_output

    def training_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        summary_labels = batch['labels']
        summary_attention_mask = batch['labels_attention_mask']
        #cl_labels = batch['cl_labels']
        #cl_attention_mask = batch['cl_labels_attention_mask']

        summary_loss, summary_logits, cl_output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=summary_attention_mask,
            labels=summary_labels,
        )

        cl_loss = self.criterion(cl_output, batch["label"])
		

        if 1 in batch["label"]:
            loss = SUM_LOSS_COEFF*summary_loss + CL_LOSS_COEFF*cl_loss*MIXTURE_COEFF
        elif 3 in batch["label"]:
            loss = SUM_LOSS_COEFF*summary_loss + CL_LOSS_COEFF*cl_loss*UNPROVEN_COEFF
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

        summary_loss, summary_logits, cl_output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=summary_attention_mask,
            labels=summary_labels,
        )


        cl_loss = self.criterion(cl_output, batch["label"])

        if 1 in batch["label"]:
            loss = SUM_LOSS_COEFF*summary_loss + CL_LOSS_COEFF*cl_loss*MIXTURE_COEFF
        elif 3 in batch["label"]:
            loss = SUM_LOSS_COEFF*summary_loss + CL_LOSS_COEFF*cl_loss*UNPROVEN_COEFF
        else:
            loss = SUM_LOSS_COEFF*summary_loss + CL_LOSS_COEFF*cl_loss

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_summary_loss", summary_loss, prog_bar=True, logger=True)
        self.log("val_classification_loss", cl_loss, prog_bar=True, logger=True)
        
        return loss

    def test_step(self, batch, batch_size):
        global model_out
        global reference
        
        _, _, cl_preds = self(input_ids=batch["text_input_ids"], 
                            attention_mask=batch["text_attention_mask"], 
                            labels=batch["labels"], 
                            decoder_attention_mask=batch['labels_attention_mask'])

        y_preds = torch.max(cl_preds, axis=1).indices.detach().cpu().tolist()
        y_true = batch["label"].detach().cpu().tolist()

        model_out = model_out + y_preds
        reference = reference + y_true

    def configure_optimizers(self):

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
        input_ids=text_encoding['input_ids'].cuda(),
        attention_mask=text_encoding['attention_mask'].cuda(),
        max_length=SUMMARY_TOKEN_LEN,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    generated_ids = generated_ids.detach().cpu().numpy()

    preds = [
            tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
    ]
    return "".join(preds)

def read_files(train_path, dev_path, test_path):
    train_df = pd.read_csv(train_path, sep="\t", encoding='utf-8') #
    val_df = pd.read_csv(dev_path, sep="\t", encoding='utf-8')
    test_df = pd.read_csv(test_path, sep="\t", encoding='utf-8')

    train_df = train_df[['claim', 'explanation', 'main_text', 'label']]
    train_df = train_df.dropna()
    train_df.columns = ['claim', 'summary', 'text', 'label']
    train_df = train_df[train_df.label != "snopes"]

    val_df = val_df[['claim', 'explanation', 'main_text', 'label']]
    val_df = val_df.dropna()
    val_df.columns = ['claim', 'summary', 'text', 'label']
    val_df = val_df[val_df.label != "National, Candidate Biography, Donald Trump,"]

    test_df = test_df[['claim', 'explanation', 'main_text', 'label']]
    test_df = test_df.dropna()
    test_df.columns = ['claim', 'summary', 'text', 'label']

parser = argparse.ArgumentParser() 
parser.add_argument('--model_name', default='google/flan-t5-large', type=str, help='model name')
parser.add_argument('--train_file', default='PUBHEALTH/train.tsv', type=str, help='train file path')
parser.add_argument('--dev_file', default='PUBHEALTH/dev.tsv', type=str, help='validation file path')
parser.add_argument('--test_file', default='PUBHEALTH/test.tsv', type=str, help='test file path')
parser.add_argument('--batch', default=4, type=int, help='batch size')
parser.add_argument('--epoch', default=3, type=int, help='number of epochs')
parser.add_argument('--hidden_size', default=128, type=int, help='hiddem dim size')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--cl_coeff', default=0.5, type=float, help='classification weight')
parser.add_argument('--summ_coeff', default=0.5, type=float, help='summarization weight')
parser.add_argument('--mixture_coeff', default=2.5, type=float, help='mixture weight')
parser.add_argument('--unproven_coeff', default=7.0, type=float, help='unproven weight')
parser.add_argument('--skip_train', default=False, type=bool, help='skip training')
args = parser.parse_args()

MODEL_NAME = args.model_name
BATCH_SIZE = args.batch
SKIP_TRAIN = args.skip_train
CL_HIDDEN_SIZE = args.hidden_size
N_EPOCHS = args.epoch
LR = args.lr
SUM_LOSS_COEFF = args.summ_coeff
CL_LOSS_COEFF = args.cl_coeff
MIXTURE_COEFF = args.mixture_coeff
UNPROVEN_COEFF = args.unproven_coeff

EXP_NAME = "model=" + MODEL_NAME + "|batch=" + str(BATCH_SIZE) + "|epoch=" + str(N_EPOCHS) + "|sum_coeff=" + str(SUM_LOSS_COEFF) + "|cl_coeff=" + str(CL_LOSS_COEFF) + "|mixture_coeff=" + str(MIXTURE_COEFF) + "|unproven_coeff=" + str(UNPROVEN_COEFF) + "|text_token_len=" + str(TEXT_TOKEN_LEN) + "|lr=" + str(LR) + "|cl_hidden_size=" + str(CL_HIDDEN_SIZE) + "|cl_dropout_prob=" + str(CL_DROPOUT_PROB)

# preprocess train-dev-test files
train_df, val_df, test_df = read_files(args.train_file, args.dev_file, args.test_file)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

data_module = NewsMTDataModule(train_df, val_df, test_df, tokenizer, BATCH_SIZE)

if not SKIP_TRAIN:
    steps_per_epoch=len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 5

    model = NewsMTModel(n_warmup_steps=warmup_steps, n_training_steps=total_training_steps)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints-pubhealth',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    progress_bar_callback = TQDMProgressBar(refresh_rate=30)

    logger = WandbLogger(project='pubhealth-flant5', name=EXP_NAME)

    trainer = pl.Trainer(
        logger=logger,
        #checkpoint_callback=checkpoint_callback,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar_callback],
        max_epochs=N_EPOCHS,
        #gpus=1,
        #progress_bar_refresh_rate=30
    )

    trainer.fit(model, data_module)

    model_path = trainer.checkpoint_callback.best_model_path
    model = NewsMTModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)


else:
    model = NewsMTModel.load_from_checkpoint(model_path)


model.freeze()
model.eval()

if SUM_LOSS_COEFF > 0:
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
if not SKIP_TRAIN:
    trainer.test(model, data_module)
else:
    data_module.setup()
    test_loader = data_module.test_dataloader()
    for batch in test_loader:
        _, _, cl_preds = model(input_ids=batch["text_input_ids"].cuda(), 
                            attention_mask=batch["text_attention_mask"].cuda(), 
                            labels=batch["labels"].cuda(), 
                            decoder_attention_mask=batch['labels_attention_mask'].cuda())

        y_preds = torch.max(cl_preds, axis=1).indices.detach().cpu().tolist()
        y_true = batch["label"].detach().cpu().tolist()

        model_out = model_out + y_preds
        reference = reference + y_true

print(confusion_matrix(reference, model_out))

print("f1-macro: ", f1_score(reference, model_out, average='macro'))
print("f1-micro: ", f1_score(reference, model_out, average='micro'))
print("f1-weighted: ", f1_score(reference, model_out, average='weighted'))
print("prec-score: ", precision_score(reference, model_out, average='weighted'))
print("recall-score: ", recall_score(reference, model_out, average='weighted'))
print("acc-score: ", accuracy_score(reference, model_out))

params = EXP_NAME + "|model_path=" +model_path
print(params)
