# ML-and-DL-for-Iris-Classification-and-Image-Age-Prediction
ncludes data preprocessing, model training, saving models to disk (logistic_model.pkl and keras_model.keras), and performing inference on user-provided input.
Iris Classification: Training and inference using Scikit-learn's Logistic Regression and a Keras neural network to classify iris flowers based on four features.



Age Group Prediction: Using a pre-trained Vision Transformer (ViT) model to predict age groups from facial images.

Prerequisites





Python Environment: Python 3.11 or compatible version.



Dependencies:

pip install numpy pandas scikit-learn tensorflow transformers torch pillow requests matplotlib joblib



Hardware: GPU (e.g., T4) recommended for faster training and inference, though CPU is sufficient.



Internet Access: Required to download the pre-trained ViT model and sample image.

Setup Instructions





Clone or download the repository containing the notebook.



Install the required dependencies using the command above.



Ensure the notebook is run in an environment like Google Colab or Jupyter with GPU support for optimal performance.



Verify that the sample image URL (https://github.com/dchen236/FairFace/raw/master/detected_faces/race_Asian_face0.jpg?raw=true) is accessible or replace it with a local image path.

Notebook Structure





Imports: Libraries for data processing, model training, and visualization.



Age Group Prediction:





Loads a pre-trained ViT model (antonioglass/vit-age-classifier).



Downloads and processes a facial image.



Performs inference and displays the predicted age group with confidence.



Iris Classification:





Data Preparation: Loads the Iris dataset and splits it into training and test sets.



Scikit-learn Model: Trains a Logistic Regression model, saves it as logistic_model.pkl, and performs inference.



Keras Model: Trains a neural network, saves it as keras_model.keras, and performs inference.



Inference: Accepts user input for four iris features and predicts the species using both models.

Running the Notebook





Open 2025_07_30.ipynb in Jupyter or Google Colab.



Execute cells sequentially to:





Load models and process the image for age prediction.



Train and save the iris classification models.



Input custom iris features (e.g., 5.1, 3.5, 1.4, 0.2) for inference.



Observe outputs, including:





Visualized age prediction with confidence score.



Predicted iris species from both models.

Outputs





Saved Models:





logistic_model.pkl: Scikit-learn Logistic Regression model.



keras_model.keras: Keras neural network model.



Visualizations: Matplotlib plot of the input image with predicted age group.



Console Outputs: Predicted class indices and names for iris classification.

Notes





The ViT model may issue a deprecation warning for ViTFeatureExtractor. This is expected and does not affect functionality.



Ensure input for iris classification follows the order: sepal length, sepal width, petal length, petal width.



For custom images in age prediction, ensure they are in RGB format and access
