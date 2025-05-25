
---

# Human Activity Recognition (HAR) using 2D CNN

This project implements a Human Activity Recognition (HAR) system using a **2D Convolutional Neural Network (CNN)** trained on the UCI HAR dataset. The trained model is deployed using **Streamlit**, allowing users to upload a `.csv` file representing a single activity sample to receive a predicted activity label.

## Features

- 2D CNN architecture for human activity recognition.
- Trained on the publicly available UCI HAR dataset.
- Streamlit web interface for live prediction.
- Accepts real user input in the form of `.csv`.
- Supports reshaping and preprocessing for 2D CNN inference.

## Project Structure

```
project/
├── UCIDataset/           # UCI HAR dataset used for training
├── har_weights.h5        # Trained CNN model weights
├── cnn_model.py          # Training script for the 2D CNN model
├── app.py                # Streamlit app for running inference
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Dataset Format

Each sample consists of:
- **128 time steps**
- **9 features** (from accelerometer and gyroscope axes)

The model expects one input sample of shape `(128, 9)` which is reshaped internally to match the 2D CNN input format: `(1, 128, 9, 1)`.

## How to Train the Model

1. Download the UCI HAR Dataset and place it inside the `UCIDataset/` folder.
2. Run the following script to train the 2D CNN and save the model:

```bash
python cnn_model.py
```

The model weights will be saved as `har_weights.h5`.

## How to Run the Web App

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Launch the Streamlit app:

```bash
streamlit run app.py
```

3. Upload a ".csv" file containing a single sample (shape: `128x9`). The model will return a predicted activity from the following classes:

```
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING
```

## File Input Format

### CSV:
- Comma-separated file with 128 rows and 9 columns.

## Requirements

Make sure the following packages are installed (listed in `requirements.txt`):

- tensorflow
- numpy
- pandas
- streamlit

## Notes

- This implementation is intended for single-sample inference, not batch predictions.
- Training code assumes access to full UCI HAR Dataset.
- Ensure uploaded files match the expected sample format, or the model will reject them.

