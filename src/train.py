import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def train_model():
    # 1. Carregar os dados já preparados (que você gerou no notebook)
    train_df = pd.read_csv('../data/train.csv')
    val_df = pd.read_csv('../data/val.csv')

    # Mapear os sentimentos para números
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    train_df['label_id'] = train_df['label'].map(label_map)
    val_df['label_id'] = val_df['label'].map(label_map)

    # 2. Carregar o tokenizer e o modelo BERTimbau
    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 3. Criar a classe do Dataset
    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                str(self.texts[idx]),
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    train_dataset = SentimentDataset(train_df['text'].values, train_df['label_id'].values, tokenizer)
    val_dataset = SentimentDataset(val_df['text'].values, val_df['label_id'].values, tokenizer)

    # 4. Função de métricas
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    # 5. Configurar o treinamento
    training_args = TrainingArguments(
        output_dir='../models/bertimbau-sentiment',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch", # Corrigido para a versão nova
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. Treinar!
    print("Starting training...")
    trainer.train()
    print("Training complete!")

if __name__ == "__main__":
    train_model()