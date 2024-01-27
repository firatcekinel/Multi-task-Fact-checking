This study primarily focuses on designing a multi-task explainable misinformation detection model. To be more specific, a fact-checking model is trained on veracity prediction and text summarization tasks simultaneously. The generated summaries are derived from evidence documents and serve as justifications for the model’s veracity prediction. Therefore, it should not be considered as a post-hoc explainability model. The contribution of the work lies in the use of multi-task learning for fact-checking and text summarization together, particularly through a new architecture including different neural models. The tasks, fact-checking and summarization, complement each other such that one does misinformation detection while the other explains the reason for the model’s decision. 

You need to install the packages listed in "requirements.txt" file to execute the codes. We also presented single-task t5-based summarization and classification models for comparison. 

Note that in this study we utilized [PUBHEALTH](https://github.com/neemakot/Health-Fact-Checking) (Kotonya and Toni, 2020), [FEVER](https://fever.ai/resources.html) (Thorne et al., 2018) and [e-FEVER](https://truthandtrustonline.com/wp-content/uploads/2020/10/TTO04.pdf) (Ash and Stammbach, 2020) datasets.

## Model Architecture

The model architecture is given in the Figure. Both summarization and classification tasks share a T5 Encoder during training. At first, the T5 Encoder encodes the claim and evidence sentences in a latent space. Afterwards, the T5 Decoder produces a summary using the T5 Encoder's representation. Simultaneously, for the veracity prediction, the encoder's output is processed by two feed-forward layers respectively. We employ the ReLU activation function and apply dropout between two linear layers and the sigmoid activation function after the second linear layer. Besides, the cross entropy loss is used for measuring summary and classification losses.

![t5-mt](https://github.com/firatcekinel/Multi-task-Fact-checking/assets/88368345/5a52174b-813d-4a2b-ba64-a7ff6de216e0)
