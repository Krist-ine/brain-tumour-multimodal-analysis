import os
import pandas as pd
import numpy as np

DATA_ROOT = "data"
os.makedirs(DATA_ROOT, exist_ok=True)

NUM_SAMPLES = 50      # number of patients
NUM_NODES = 10        # number of nodes per graph
NUM_CLASSES = 4       # tumor types

# Generate random node features
data = np.random.rand(NUM_SAMPLES, NUM_NODES)
labels = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))

df = pd.DataFrame(data, columns=[f"node_feat_{i}" for i in range(NUM_NODES)])
df['tumor_type'] = labels

# Save train and test CSVs
df.to_csv(os.path.join(DATA_ROOT, "synthetic_clinical_train.csv"), index=False)
df.to_csv(os.path.join(DATA_ROOT, "synthetic_clinical_test.csv"), index=False)

print("Synthetic clinical CSVs created successfully!")
