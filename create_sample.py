import pandas as pd
import os
import numpy as np

CHUNKSIZE = 100000
SAMPLE_FRACTION = 0.01
INPUT_FILE = 'train/train.csv'
OUTPUT_FILE = 'train/train_stratified_sample.csv'

reader = pd.read_csv(INPUT_FILE, chunksize=CHUNKSIZE)
header_written = False

for i, chunk in enumerate(reader):
    # Perform stratified sampling on the chunk
    # We group by 'label' and take a fraction of each group
    try:
        sampled_chunk = chunk.groupby('label').sample(frac=SAMPLE_FRACTION, 
                                                      replace=False, 
                                                      random_state=42)
    except ValueError:
        # if a group is too small for the fraction
        # We'll just take 1 sample from each group in that rare case.
        sampled_chunk = chunk.groupby('label').sample(n=1, 
                                                      replace=False, 
                                                      random_state=42)

    # Write the first chunk with a header, append for all others
    if not header_written:
        sampled_chunk.to_csv(OUTPUT_FILE, index=False)
        header_written = True
    else:
        sampled_chunk.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    
    print(f"Processed chunk {i+1}...")
