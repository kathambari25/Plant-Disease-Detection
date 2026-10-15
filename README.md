# Plant Disease Detection Using CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) to identify plant diseases from leaf images using deep learning.

The model is trained on the PlantVillage dataset and classifies images into 38 plant disease and healthy-leaf categories.

## Technologies Used
- Python
- TensorFlow
- Keras
- CNN
- Streamlit
- NumPy
- Pillow

## Features
- Upload a plant leaf image.
- Predict the possible plant disease.
- Display prediction confidence.
- Show the top 3 predictions.
- Classify 38 plant disease and healthy-leaf categories.

## Model Details
- Model: Convolutional Neural Network (CNN)
- Image Size: 128 × 128
- Number of Classes: 38
- Validation Accuracy: 92.66%
- Model File: `plant_disease_model.keras`

## Dataset
PlantVillage dataset containing plant leaf images of healthy plants and plants affected by different diseases.

## How to Run

1. Install the required libraries:

   `pip install -r requirements.txt`

2. Run the application:

   `streamlit run app.py`

## Future Enhancements
- Add more plant disease categories.
- Improve prediction accuracy.
- Provide disease prevention and management information.

## Author
Kathambari
