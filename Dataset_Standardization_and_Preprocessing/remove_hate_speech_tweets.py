import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "savasy/bert-base-turkish-sentiment-cased"  # Fine-tuned Turkish model for sentiment
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to classify text
def classify_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:  # Handle empty or invalid input
        return -1  # Mark invalid inputs as -1
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class


file_paths = [
    'datasets/table3/combined_cleaned_table3.csv'
]


dataframes = [pd.read_csv(file) for file in file_paths]
data = pd.concat(dataframes, ignore_index=True)


if 'Text' not in data.columns:
    raise ValueError("The dataset must contain a 'Text' column.")

# Classify each row and filter out hate speech (assuming class 1 is 'hate speech')
data['is_hate'] = data['Text'].apply(lambda x: classify_text(x))
filtered_data = data[data['is_hate'] == 0]  # Keep only non-hate speech

# Save the filtered dataset
filtered_file_path = 'datasets/filtered/filtered_table_3_data.csv'
filtered_data.to_csv(filtered_file_path, index=False)
print(f"Filtered dataset saved to {filtered_file_path}")
