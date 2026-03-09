import cv2
import sys
import os
import pickle
import numpy as np
from deepface import DeepFace

# Define constants
EMBEDDINGS_FILE = "face_embeddings.pkl"
# Standard Cosine Distance Threshold for FaceNet is ~0.40. Smaller = stricter.
THRESHOLD = 0.40

def cosine_distance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def predict_identity(img_path):
    if not os.path.exists(EMBEDDINGS_FILE):
        print("Model untrained. Please run train.py first.")
        return None

    with open(EMBEDDINGS_FILE, "rb") as f:
        centroids = pickle.load(f)

    print("Extracting facial features using FaceNet...")
    try:
        # Load the test image and extract features
        result = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
        if len(result) == 0:
            print("No face detected.")
            return None
            
        emb = result[0]["embedding"]
        # Normalize the test embedding
        emb = emb / np.linalg.norm(emb)
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

    # Compare with known centroids to find nearest match
    closest_person = "Unknown"
    min_dist = float('inf')

    for person, centroid in centroids.items():
        dist = cosine_distance(emb, centroid)
        print(f"Cosine Distance to {person}: {dist:.4f}")
        
        if dist < min_dist:
            min_dist = dist
            # Must strictly be under the FaceNet cosine threshold
            if min_dist < THRESHOLD:
                closest_person = person

    if closest_person != "Unknown":
        print(f"\nIdentity Recognized: '{closest_person}'.")
    else:
        print("\nIdentity: UNKNOWN. Person restricted (Access Denied).")

    # Read original image for drawing bounding info
    original_img = cv2.imread(img_path)
    if original_img is None:
        print("Could not read original image for output.")
        return closest_person, None

    text = f"ID: {closest_person}"
    color = (0, 0, 255) # Red for restricted
    if closest_person != "Unknown":
        color = (0, 255, 0) # Green for allowed

    cv2.putText(original_img, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    output_filename = f"result_{os.path.basename(img_path)}"
    cv2.imwrite(output_filename, original_img)
    
    return closest_person, original_img

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_test_image>")
        sys.exit(1)

    test_image_path = sys.argv[1]
    predict_identity(test_image_path)
