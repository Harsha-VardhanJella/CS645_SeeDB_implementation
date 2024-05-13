from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
census_income = fetch_ucirepo(id=20) 
  
X = census_income.data.features 
y = census_income.data.targets 
  
combined_data = pd.concat([X, y], axis=1)

num_files = 6
chunk_size = len(combined_data) // num_files

#Splitting the data into 6 different partitions and writing it to a file
for i in range(num_files):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size if i < num_files - 1 else len(combined_data)
    chunk = combined_data.iloc[start_idx:end_idx]
    chunk.to_csv(f'file_{i + 1}.txt', sep=',', index=False)

#Writing the entire data to a file
combined_data.to_csv('combined_data.txt', sep=',', index=False) 