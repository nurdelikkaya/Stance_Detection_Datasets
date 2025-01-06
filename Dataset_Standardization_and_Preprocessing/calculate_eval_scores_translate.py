import pandas as pd
import sacrebleu
from nltk.translate.meteor_score import meteor_score
import csv  # Import csv module for quoting options

# Clean sentences (remove unnecessary quotes, spaces, etc.)
def clean_sentences(sentences):
    return [sentence.strip().strip('"').strip("'") for sentence in sentences]

# Function to read and clean a single column from CSV
def read_sentences(file_path):
    df = pd.read_csv(file_path)
    return clean_sentences(df.iloc[:, 0].dropna().tolist())

# Compute METEOR scores
def calculate_meteor(translations, references):
    scores = [meteor_score([ref], hyp) for ref, hyp in zip(references, translations)]
    return sum(scores) / len(scores)

# Tokenize sentences before computing METEOR
def tokenize_sentences(sentences):
    return [sentence.split() for sentence in sentences]



# Function to remove quotes
def remove_start_end_quotes(file_path, output_file):
    df = pd.read_csv(file_path)
    # Apply a function to strip starting and ending quotes from all cells
    df = df.applymap(lambda x: x[1:-1] if isinstance(x, str) and x.startswith('"') and x.endswith('"') else x)
    # Save the cleaned dataframe
    df.to_csv(output_file, index=False)

# Use the updated function
remove_start_end_quotes("translated_tryout.csv", "translated_tryout_cleaned.csv")
remove_start_end_quotes("headlines_only2.csv", "headlines_only_cleaned.csv")
# Load and clean datasets
reference_file = "example.csv"
helsinki_file = "translated_tryout.csv"
google_file = "headlines_only2.csv"

references = read_sentences(reference_file)
helsinki_translations = read_sentences(helsinki_file)
google_translations = read_sentences(google_file)

# Ensure all lists are the same length
min_len = min(len(references), len(helsinki_translations), len(google_translations))
references = references[:min_len]
helsinki_translations = helsinki_translations[:min_len]
google_translations = google_translations[:min_len]

# Compute BLEU scores
bleu_helsinki = sacrebleu.corpus_bleu(helsinki_translations, [references])
bleu_google = sacrebleu.corpus_bleu(google_translations, [references])

# Compute ChrF scores
chrf_helsinki = sacrebleu.corpus_chrf(helsinki_translations, [references])
chrf_google = sacrebleu.corpus_chrf(google_translations, [references])

# Tokenize sentences for METEOR
references = tokenize_sentences(references)
helsinki_translations = tokenize_sentences(helsinki_translations)
google_translations = tokenize_sentences(google_translations)

# Compute METEOR scores
meteor_helsinki = calculate_meteor(helsinki_translations, references)
meteor_google = calculate_meteor(google_translations, references)

# Print BLEU scores
print(f"BLEU Score for Helsinki Translations: {bleu_helsinki.score:.2f}")
print(f"BLEU Score for Google Translations: {bleu_google.score:.2f}")

# Print ChrF scores
print(f"ChrF Score for Helsinki Translations: {chrf_helsinki.score:.2f}")
print(f"ChrF Score for Google Translations: {chrf_google.score:.2f}")

# Print METEOR scores
print(f"METEOR Score for Helsinki Translations: {meteor_helsinki:.4f}")
print(f"METEOR Score for Google Translations: {meteor_google:.4f}")
