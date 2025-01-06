from transformers import pipeline
import pandas as pd
from memory_profiler import profile

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-tr")

# Read the dataset
df = pd.read_csv('headlines_only.csv')

# Function to batch translate text
@profile
def batch_translate(text_list, batch_size=16):
    translated_texts = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        results = pipe(batch, max_length=512)
        translations = [res['translation_text'] for res in results]
        translated_texts.extend(translations)
    return translated_texts

## Translate every column
for column in df.columns:
    if df[column].dtype == object:  # Only translate text columns
        print(f"Translating column: {column}")
        text_data = df[column].fillna("").tolist()  # Handle missing values
        df[column] = batch_translate(text_data, batch_size=16)

# Save the fully translated dataset to a new CSV
df.to_csv('translated_example.csv', index=False)

# Print success message
print("Translation complete. The new file 'fully_translated_stances.csv' has been saved.")