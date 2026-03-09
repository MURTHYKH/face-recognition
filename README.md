# Face Recognition Project

This project uses a pre-trained CNN (MobileNetV2 from ImageNet) to extract facial embeddings and restrict access to unknown persons, while performing Age Estimation using DeepFace.

## Features:
1. **Identity Recognition**: Uses an ImageNet-trained CNN to generate face embeddings. Compares a target image against the people in the `images` folder using Euclidean distance.
2. **Access Restriction**: If the person's embedding distance is above a critical threshold, it classifies them as `Unknown` and restricts access.
3. **Age Prediction**: Uses the DeepFace library to predict age intelligently, as estimating age from limited single-person datasets is prone to failure without massive demographic training data.

## Project Structure:
- `images/`: The folder containing subdirectories of known people (e.g., `images/murthy/`). Add as many images per person as possible.
- `train.py`: The script used to train the recognizer. It loops through the `images` folder, computes features for each person, and saves the centroids to `face_embeddings.pkl`.
- `predict.py`: The script to evaluate a single image. It extracts features to verify identity and estimates the predicted age.

## Usage:

### 1. Training the Recognizer
Whenever you add or remove images in the `images` folder, re-run the training script to update the expected person centroids.

```bash
python train.py
```

This will generate a `face_embeddings.pkl` file containing the recognized people.

### 2. Predicting Age & Restricting Persons
To evaluate an unseen test image, use `predict.py` with the path to the test image:

```bash
python predict.py "path_to_test_image.jpg"
```

The script will output the predicted identity and age (or report "Unknown").

**Note on Thresholds:**
The `predict.py` script has a `THRESHOLD` constant (currently 8.5). If legitimate known people are rejected as "Unknown", increase this number. If random strangers are incorrectly recognized as known people, decrease it.
