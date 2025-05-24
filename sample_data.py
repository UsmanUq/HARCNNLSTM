import numpy as np
import pandas as pd

# Paths to one sample's data (first row from each of the 9 signals)
signals = [
    'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt',
    'body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
    'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
]

data_dir = 'UCIDataset/train/Inertial Signals/'  # adjust if needed
sample_index = 3

sample_data = []

for signal_file in signals:
    data = np.loadtxt(data_dir + signal_file)
    sample_data.append(data[sample_index])

# Transpose from (9, 128) → (128, 9)
sample_matrix = np.array(sample_data).T

# Save to CSV
df = pd.DataFrame(sample_matrix)
df.to_csv("sample_input.csv", index=False, header=False)

print("✅ Sample saved as sample_input.csv with shape:", df.shape)
