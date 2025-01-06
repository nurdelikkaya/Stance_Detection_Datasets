import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'train_stances.csv'
df = pd.read_csv(file_path)

# Drop rows where 'Stance' column is 'unrelated'
df = df[df['Stance'] != 'unrelated']

# Map the 'Stance' labels to numerical values
label_mapping = {'agree': 1, 'disagree': -1, 'discuss': 0}
df['Stance'] = df['Stance'].map(label_mapping)

# Perform stratified splitting into 70-15-15
train, temp = train_test_split(df, test_size=0.3, stratify=df['Stance'], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['Stance'], random_state=42)

# Save the splits into separate CSV files (optional)
train.to_csv('fnc_train.csv', index=False)
val.to_csv('fnc_validation.csv', index=False)
test.to_csv('fnc_test.csv', index=False)

print(f"Training set size: {len(train)}")
print(f"Validation set size: {len(val)}")
print(f"Test set size: {len(test)}")
