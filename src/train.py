# src/train.py

import torch
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
from datasets import load_metric

# Import from our other project files
from config import Config
from data import get_tokenized_dataset

def main():
    config = Config()
    
    # --- 1. Load and prepare the dataset ---
    print("Loading and tokenizing dataset...")
    tokenized_dataset, tokenizer = get_tokenized_dataset()
    
    # --- 2. Initialize the model ---
    print("Initializing model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_CHECKPOINT)
    
    # --- 3. Set up training arguments ---
    # These arguments define all aspects of the training process
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        weight_decay=config.WEIGHT_DECAY,
        save_total_limit=3,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        predict_with_generate=True,
        fp16=True if torch.cuda.is_available() else False, # Use mixed precision if a GPU is available
        push_to_hub=False,
    )
    
    # --- 4. Define evaluation metrics ---
    # The ROUGE score is the standard metric for summarization
    metric = load_metric("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent.strip() for sent in pred.split()) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent.strip() for sent in label.split()) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    # --- 5. Create the Trainer ---
    # The Trainer object orchestrates the entire training process
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 6. Start Training ---
    print("Starting training...")
    # trainer.train() # This is the command to start the actual training
    print("\nSetup complete. The project is ready to be trained by running `trainer.train()`.")
    
    # In a real run, you would save the model after training
    # trainer.save_model(f"{config.OUTPUT_DIR}/best_model")

if __name__ == "__main__":
    main()