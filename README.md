# Land Classification Web App

A minimalist web application for classifying land types in satellite/aerial images using deep learning.

## Features

- Upload satellite/aerial images
- Automatic image preprocessing
- Real-time classification using a pre-trained CNN model
- Simple and intuitive user interface
- Shows classification results with confidence scores

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. **Option A - Use Pre-trained Model (Recommended):**
   
   The repository includes a pre-trained model (`model.h5`) ready to use. Skip to step 3.

   **Option B - Train Your Own Model:**
   
   ```bash
   python train_model.py
   ```
   
   *Note: This will download the EuroSAT dataset (~2GB) and train a new model.*

3. Start the Flask server:

```bash
python app.py
```

4. Open your web browser and navigate to `http://localhost:5000`

## Dataset Information

**Important**: The training dataset (27,000 images from EuroSAT) is **NOT included** in this repository to keep it lightweight. The dataset will be automatically downloaded when you run `train_model.py`.

- **For immediate use**: A pre-trained model is included
- **For training**: Run the training script to download the dataset automatically
- **Dataset size**: ~2GB (27,000 RGB images across 10 land-use classes)

## Usage

1. Click the "Choose Image" button to select a satellite/aerial image
2. The selected image will be displayed in the preview area
3. Click the "Classify" button to process the image
4. The results will show the predicted land type and confidence score

## Model Details

- Uses a Convolutional Neural Network (CNN) trained on the EuroSAT dataset
- Input image size: 64x64 pixels
- Output classes: 10 different land types including agricultural, urban, water, forest, etc.
- Model architecture: 3 convolutional layers with max pooling, followed by dense layers

## Technical Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python Flask
- Machine Learning: TensorFlow/Keras
- Image Processing: Pillow

## Note

The first run will download the EuroSAT dataset (approximately 2GB) and train the model, which may take some time depending on your system's specifications. 