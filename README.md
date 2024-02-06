This study primarily focuses on designing a multi-task explainable misinformation detection model. To be more specific, a fact-checking model is trained on veracity prediction and text summarization tasks simultaneously. The generated summaries are derived from evidence documents and serve as justifications for the model’s veracity prediction. Therefore, it should not be considered as a post-hoc explainability model. The contribution of the work lies in the use of multi-task learning for fact-checking and text summarization together, particularly through a new architecture including different neural models. The tasks, fact-checking and summarization, complement each other such that one does misinformation detection while the other explains the reason for the model’s decision. 

You need to install the packages listed in "requirements.txt" file to execute the codes.  

Note that in this study we utilized [PUBHEALTH](https://github.com/neemakot/Health-Fact-Checking) (Kotonya and Toni, 2020), [FEVER](https://fever.ai/resources.html) (Thorne et al., 2018) and [e-FEVER](https://truthandtrustonline.com/wp-content/uploads/2020/10/TTO04.pdf) (Ash and Stammbach, 2020) datasets.

## Model Architecture

The model architecture is given in the Figure. Both summarization and classification tasks share a T5 Encoder during training. At first, the T5 Encoder encodes the claim and evidence sentences in a latent space. Afterwards, the T5 Decoder produces a summary using the T5 Encoder's representation. Simultaneously, for the veracity prediction, the encoder's output is processed by two feed-forward layers respectively. We employ the ReLU activation function and apply dropout between two linear layers and the sigmoid activation function after the second linear layer. Besides, the cross entropy loss is used for measuring summary and classification losses.

![t5-mt](https://github.com/firatcekinel/Multi-task-Fact-checking/assets/88368345/5a52174b-813d-4a2b-ba64-a7ff6de216e0)

## Execution

You can execute the scripts with the following arguments:

```
python t5_multitask_efever.py \
--model_name "google/flan-t5-large" \
--train_file "FEVER/efever_train.tsv" \
--dev_file "FEVER/dev_train.tsv" \
--test_file "FEVER/test_train.tsv" \
--batch 4 \ 
--epoch 3 \ 
--hidden_size 64 \
--lr 2e-5 \
--cl_coeff 0.9 \
--summ_coeff 0.1 \
--useUncertaintyLoss False \
--skip_train False
```

or 

```
python t5_multitask_pubhealth.py \
--model_name "google/flan-t5-large" \
--train_file "PUBHEALTH/train.tsv" \
--dev_file "PUBHEALTH/dev.tsv" \
--test_file "PUBHEALTH/test.tsv" \
--batch 4 \ 
--epoch 3 \ 
--hidden_size 128 \
--lr 1e-4 \
--cl_coeff 0.5 \
--summ_coeff 0.5 \
--mixture_coeff 2.5 \
--unproven_coeff 7.0 \
--skip_train False
```

## Citation
Please cite the paper as follows if you find the study useful.