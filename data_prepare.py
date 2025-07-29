import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn import preprocessing
from tqdm import tqdm
import time
import os
dataset_path = []

for dirname, _, filenames in os.walk('dataset/FlowBasedFeatures/DDoS'):
    for filename in filenames:
        if filename.endswith('.csv'):
            dfp = os.path.join(dirname, filename)
            dataset_path.append(dfp)

#print(dataset_path)

cols = list(pd.read_csv(dataset_path[1], nrows=1))

def load_file(path):
    data = pd.read_csv(path)
    # Gán nhãn dựa theo tên thư mục cha chứa file CSV
    label = os.path.basename(os.path.dirname(path))
    data['Label'] = label
    return data
    
# Load files with progress bar
print("\nLoading CSV files...")
processed_count = 0
skipped_count = 0
samples_list = []

for dfp in tqdm(dataset_path, desc="Processing files", unit="file"):
    try:
        data = load_file(dfp)
        if data is not None:
            samples_list.append(data)
            processed_count += 1
        else:
            skipped_count += 1
        time.sleep(0.1)  # Small delay to see progress
    except Exception as e:
        print(f"Unexpected error processing {dfp}: {str(e)}")
        skipped_count += 1
        continue

print(f"\nProcessing complete. Successfully processed {processed_count} files, skipped {skipped_count} files.")

if processed_count == 0:
    print("Error: No valid files were processed. Please check your dataset files.")
    exit(1)

if not samples_list:
    print("\nError: No valid files with 'Label' column were found.")
    exit(1)

print("\nCombining data...")
samples = pd.concat(samples_list, ignore_index=True)
print(f"Total rows after combining: {len(samples)}")

import matplotlib.pyplot as plt

# Count the occurrences of each label
label_counts = samples['Label'].value_counts()

# Create a bar plot to visualize the label counts
plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar')
plt.title('Comparison of Label Column')
plt.xlabel('Label')
plt.ylabel('Count')
#plt.show()
plt.savefig('label_distribution.png')

grouped = samples.groupby('Label')

output_dir = 'class_split'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for label, group in grouped:
    filename = os.path.join(output_dir, f"{label}.csv")
    group.to_csv(filename, index=False)

import pandas as pd
import os

directory = './class_split/'

files = [
    'DDoS ACK Fragmentation.csv', 'DDoS ICMP Flood.csv', 
    'DDoS-HTTP Flood.csv', 'DDoS-ICMP_Fragmentation.csv'
]

combined_data = pd.DataFrame()

for file in files:
    file_path = os.path.join(directory, file)
    data = pd.read_csv(file_path)
    sample_data = data.sample(n=500, random_state=1)
    combined_data = pd.concat([combined_data, sample_data], ignore_index=True)

combined_data.to_csv('./combined_data.csv', index=False)
