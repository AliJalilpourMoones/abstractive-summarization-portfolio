# src/data.py

from datasets import load_dataset
from transformers import AutoTokenizer
from config import Config

def get_tokenized_dataset():
    """
    Loads the XSum dataset, tokenizes it, and prepares it for training.
    """
    config = Config()
    
    # Load the dataset from the Hugging Face Hub
    dataset = load_dataset(config.DATASET_NAME)
    
    # Load the tokenizer for our chosen model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)
    
    def preprocess_function(examples):
        # The prefix is recommended for T5 models, but is a good practice for others too
        prefix = "summarize: "
        
        # Prepare inputs and targets
        inputs = [prefix + doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs, max_length=config.MAX_INPUT_LENGTH, truncation=True)
        
        # Tokenize the summaries (labels)
        labels = tokenizer(text_target=examples["summary"], max_length=config.MAX_TARGET_LENGTH, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    # Apply the preprocessing to the entire dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset, tokenizer