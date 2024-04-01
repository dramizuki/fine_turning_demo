import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer
import tensorflow as tf
from transformers import TrainingArguments, BertForSequenceClassification
from transformers import Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from transformers import AutoModelForSequenceClassification
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


data_root = './csv'

train_df = pd.read_csv(f'{data_root}/knbc-train.csv', index_col=0)
test_df = pd.read_csv(f'{data_root}/knbc-test.csv', index_col=0)

tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

train_encodings = tokenizer(train_df['sentence'].to_list(), truncation=True, padding=True)
test_encodings = tokenizer(test_df['sentence'].to_list(), truncation=True, padding=True)

label2index = {
    'Kyoto': 0,
    'Keitai': 1,
    'Gourmet': 2,
    'Sports': 3,
}

train_labels = torch.tensor([label2index[target] for target in train_df['target'].to_list()])
test_labels = torch.tensor([label2index[target] for target in test_df['target'].to_list()])

train_dataset = Dataset(train_encodings, train_labels)
test_dataset = Dataset(test_encodings, test_labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)


model = AutoModelForSequenceClassification.from_pretrained(
    'cl-tohoku/bert-base-japanese-whole-word-masking',
    num_labels = 4,
)
model.cuda()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

print(trainer.evaluate())

model_path = './test_model'
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)
