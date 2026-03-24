# HDR
This project is a Handwritten Digit Recognizer built with a sophisticated "Online Learning" architecture. Unlike standard AI apps that remain static after training, this system allows users to correct the AI in real-time, effectively "teaching" the model as it is being used.

# ## Project Documentation: Handwritten Digit Recognizer

## ## 1. Overview
This project is an interactive **MNIST Digit Classifier** that utilizes a Multi-layer Perceptron (MLP) neural network. It features a "Formal Inference Lab" where users can draw digits on a digital canvas and receive real-time predictions. A key highlight of this system is its **Online Learning** capability, which allows the model to be updated instantly when a user provides a correction.

---

## ## 2. File Directory
The following files constitute the core of the system:

* **`app.py`**: The Streamlit web application providing the user interface, image processing logic, and model update triggers.
* **`train_model.py`**: A standalone script used to fetch the MNIST dataset ($70,000$ images) and perform the initial model training.
* **`mnist_model.joblib`**: The serialized model file (the "brain") that stores the trained weights of the neural network.
* **`requirements.txt`**: A configuration file listing all necessary Python dependencies.

---

## ## 3. Technical Specifications

### ### AI Architecture
The model is built using `scikit-learn`'s `MLPClassifier` with the following parameters:
* **Hidden Layers**: Two layers consisting of $128$ and $64$ neurons.
* **Activation**: ReLU (Rectified Linear Unit).
* **Solver**: Adam (an optimization solver).
* **Input Dimensions**: $784$ features, representing a flattened $28 \times 28$ pixel image.

### ### Image Processing Pipeline
To ensure high accuracy, `app.py` processes raw canvas drawings using **OpenCV**:
1.  **Grayscale Conversion**: Converts RGBA canvas data to a single-channel grayscale image.
2.  **Bounding Box Cropping**: Locates the drawn digit and crops the image to its edges.
3.  **Padding & Squaring**: Centers the digit within a square frame to maintain aspect ratio.
4.  **Resizing**: Scales the result to $28 \times 28$ pixels to match the original training data.

---

## ## 4. Installation & Usage

### ### Step 1: Environment Setup
Install the required libraries using pip:
```bash
pip install -r requirements.txt
```
*Note: Dependencies include `streamlit`, `opencv-python`, `scikit-learn`, `numpy`, and `joblib`.*

### ### Step 2: Model Initialization
If the `.joblib` file is not present, run the training script to generate the model:
```bash
python train_model.py
```
This script downloads the MNIST dataset, trains the model for up to $60$ iterations, and saves it.

### ### Step 3: Launching the App
Start the interactive dashboard:
```bash
streamlit run app.py
```

---

## ## 5. Online Learning Loop
The system implements a feedback loop for continuous improvement:
* **Prediction**: The model outputs a digit and a confidence percentage.
* **Correction**: If the prediction is incorrect, the user selects the "Actual Digit" and clicks **"Commit Correction"**.
* **Partial Fit**: The app calls `model.partial_fit()`, which performs an incremental update to the model's weights using the new sample.
* **Persistence**: The updated model is immediately re-saved to the disk.

---
