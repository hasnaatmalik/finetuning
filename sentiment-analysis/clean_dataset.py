import pandas as pd
import csv
rows = []
with open('sentiment-analysis.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader) 
    print(f"Header: {header}")
    
    for i, row in enumerate(reader):
        if i < 3: 
            print(f"Row {i}: {row} (length: {len(row)})")
        
        if row:  
            if len(row) >= 2:
                text = row[0].strip().strip('"').strip("'")
                sentiment = row[1].strip().strip('"').strip("'")
                rows.append({'text': text, 'sentiment': sentiment})
            elif len(row) == 1 and ',' in row[0]:
                parts = row[0].split(',', 1)
                if len(parts) >= 2:
                    text = parts[0].strip().strip('"').strip("'")
                    sentiment = parts[1].split(',')[0].strip().strip('"').strip("'")
                    rows.append({'text': text, 'sentiment': sentiment})

print(f"\nTotal rows parsed: {len(rows)}")

if len(rows) == 0:
    print("ERROR: No rows were parsed. Please check the file format.")
else:
    cleaned_df = pd.DataFrame(rows)
    
    print(f"DataFrame created with shape: {cleaned_df.shape}")
    print(f"Columns: {cleaned_df.columns.tolist()}")
    cleaned_df = cleaned_df.drop_duplicates()
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df[(cleaned_df['text'] != '') & (cleaned_df['sentiment'] != '')]
    cleaned_df.to_csv('sentiment-analysis-cleaned.csv', index=False)
    
    print(f"\n✓ Cleaned CSV created successfully!")
    print(f"Total rows: {len(cleaned_df)}")
    print(f"\nSentiment distribution:")
    print(cleaned_df['sentiment'].value_counts())
    print(f"\nFirst few rows:")
    print(cleaned_df.head())