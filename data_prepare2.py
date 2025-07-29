
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
