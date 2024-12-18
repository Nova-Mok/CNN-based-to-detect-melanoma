# CNN Model for Melanoma Detection

> A convolutional neural network (CNN) model to accurately detect melanoma, a deadly form of skin cancer. The project leverages TensorFlow and Keras to build and train the model, aiming to aid dermatologists by automating the detection process from images.

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Implementation Steps](#implementation-steps)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

## General Information
- **Background**: Melanoma accounts for 75% of skin cancer deaths. Early detection is crucial for effective treatment.
- **Objective**: To build a CNN model that evaluates images and alerts dermatologists about the presence of melanoma, reducing manual diagnostic efforts.
- **Dataset**: ISIC dataset containing 2,239 training images and 118 testing images distributed across 9 skin cancer types:
  - Actinic keratosis
  - Basal cell carcinoma
  - Dermatofibroma
  - Melanoma
  - Nevus
  - Pigmented benign keratosis
  - Seborrheic keratosis
  - Squamous cell carcinoma
  - Vascular lesion

## Technologies Used
- Python 3.x
- TensorFlow 2.x
- Keras API
- Google Colab (GPU Runtime)
- Augmentor (for data augmentation)
- Matplotlib (for visualization)

## Implementation Steps
1. **Data Loading**:
   - Used `tf.keras.preprocessing.image_dataset_from_directory` to load training and validation datasets with an 80-20 split.

2. **Data Preprocessing**:
   - Resized all images to 180x180 pixels.
   - Normalized pixel values to [0, 1] using `Rescaling`.

3. **Visualization**:
   - Visualized sample images from each class.

4. **Model Building**:
   - Constructed a CNN with:
     - Three convolutional layers followed by max-pooling layers.
     - A dense layer with 128 neurons and a dropout layer (0.5) to reduce overfitting.
     - Output layer with 9 neurons (softmax activation).

5. **Training**:
   - Trained the model for 20 epochs using:
     - Optimizer: Adam
     - Loss Function: Sparse Categorical Crossentropy
     - Metrics: Accuracy

6. **Data Augmentation**:
   - Applied random flips, rotations, and zooms using `RandomFlip`, `RandomRotation`, and `RandomZoom`.
   - Augmented class-imbalanced data using Augmentor to ensure a minimum of 500 samples per class.

7. **Retraining**:
   - Re-trained the model for 30 epochs on the augmented dataset.

8. **Evaluation**:
   - Plotted training and validation accuracy/loss curves.
   - Observed improved performance and reduced overfitting with data augmentation.

## Conclusions
- The initial model showed signs of overfitting, evident from the disparity between training and validation accuracies.
- Data augmentation effectively mitigated overfitting, improving generalization.
- Handling class imbalance significantly boosted model performance, ensuring fair representation across all classes.
- The final model achieved high accuracy in detecting melanoma and other skin cancer types.

## Acknowledgements
- **Dataset Source**: ISIC Dataset
- **References**:
  - TensorFlow documentation
  - Augmentor library documentation
  - Tutorials on data augmentation and CNN model building

## Contact
Created by [@Novamok] - feel free to contact me for further details or collaboration!

## Files with answer is 