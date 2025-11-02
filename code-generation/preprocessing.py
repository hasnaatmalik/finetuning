import pandas as pd
import json
import re
from pathlib import Path

# Input/output paths
input_path = "spoc-train.tsv"
output_path = "train.json"

# 1. Load dataset
df = pd.read_csv(input_path, sep='\t')

# 2. Basic cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\s+', ' ', str(text).strip())
    return text

df['text'] = df['text'].apply(clean_text)
df['code'] = df['code'].apply(clean_text)

# 3. Combine rows belonging to the same function (same subid)
# Some functions are split across multiple lines
grouped = df.groupby('subid').agg({
    'text': ' '.join,
    'code': '\n'.join
}).reset_index()

# 4. Format for training (pseudo-code → code)
train_samples = []
for _, row in grouped.iterrows():
    if row['text'].strip() and row['code'].strip():
        sample = {
            "input_text": f"Pseudo-code: {row['text']}\nPython code:",
            "target_text": row['code']
        }
        train_samples.append(sample)

# 5. Save cleaned data
with open(output_path, "w") as f:
    json.dump(train_samples, f, indent=2)

print(f"✅ Preprocessed dataset saved as {output_path}")
print(f"Total samples: {len(train_samples)}")
print("Example:\n", json.dumps(train_samples[0], indent=2))
