# Deep-Learning# Deep Learning Project: Microrobot Pose & Depth Estimation

This repository documents a deep learning project aimed at estimating microrobot pose (classification) and depth (regression) from images. All development and experiments have been conducted in Google Colab for convenience and GPU support.

## Project Overview

### Task 1: Data Preparation & Preprocessing
- **Loading & Exploring the Dataset:**  
  Extract the dataset from a provided ZIP file, parse `Label.txt` for labels, and verify image-label consistency.
- **Exploratory Data Analysis (EDA):**  
  Visualize sample images, examine label distributions for pose classes and depth values, and identify any data imbalances.
- **Normalization & Splitting:**  
  Normalize image pixels (rescale to [0,1]) and standardize depth values.  
  Perform stratified splits into training, validation, and test sets.
- **Optional Data Augmentation:**  
  Apply operations like rotation, shifts, and flips to increase training data diversity.

### Task 2: Pose Estimation (Classification)
- **Models Implemented:**
  1. **Simple CNN:** A baseline architecture to establish initial performance.
  2. **Deeper CNN:** A more complex model with additional convolutional layers for richer feature extraction.
  3. **MobileNetV2 (Transfer Learning):** A pre-trained model leveraged as a feature extractor, with custom layers for final classification.

- **Evaluation Metrics:**  
  Accuracy, Precision, Recall, F1-Score, and Confusion Matrix are used to evaluate model performance.

- **Current Results:**  
  MobileNetV2 outperforms the Simple and Deeper CNN architectures, yielding the highest accuracy and stable classification metrics.

### Task 3: Depth Estimation (Regression)
- **Models Implemented:**
  1. **Simple CNN (Regression):** A straightforward CNN adapted for predicting continuous depth values.
  2. **Deeper CNN (Regression):** A more advanced CNN aiming to capture finer details for depth estimation.
  3. **MobileNetV2 (Regression):** MobileNetV2-based model, using it as a feature extractor and adding a regression head for depth prediction.

- **Evaluation Metric:**  
  RMSE (Root Mean Squared Error) is employed to quantify depth estimation performance.

- **Current Results:**  
  MobileNetV2 again achieves the lowest RMSE, outperforming both the Simple and Deeper CNN models in depth estimation.

### Next Steps (Task 4 - Hyperparameter Tuning)
The next phase will involve hyperparameter tuning. Various parameters (e.g., learning rates, number of dense units, dropout rates, and possibly partial fine-tuning of MobileNetV2 layers) will be explored using Keras Tuner to further improve upon the already strong MobileNetV2 results.

## Repository Structure
- `data/`: Contains the dataset (after extraction).
- `notebooks/`:  
  - `Task1_DataPrep.ipynb`: Data loading, EDA, normalization, splitting, and augmentation steps.  
  - `Task2_PoseEstimation.ipynb`: Pose classification with Simple CNN, Deeper CNN, and MobileNetV2.  
  - `Task3_DepthEstimation.ipynb`: Depth regression using Simple CNN, Deeper CNN, and MobileNetV2.
- `models/`: Directory to store trained model weights (`.h5` files).
- `README.md`: This documentation file.

## Running the Code
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/YourRepoName.git
2. **Open in Google Colab:**
  •	Upload the .ipynb notebooks from notebooks/ to Colab or open them via a GitHub link.
	•	Ensure the Colab runtime is set to GPU for faster training.

## Dependencies
	•	Python 3.7+
	•	TensorFlow/Keras
	•	NumPy, Pandas, Matplotlib, Seaborn
	•	scikit-learn
