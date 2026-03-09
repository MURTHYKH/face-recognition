import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace

# Define paths
IMAGE_DIR = "images"
EMBEDDINGS_FILE = "face_embeddings.pkl"

def train_embeddings():
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory '{IMAGE_DIR}' not found.")
        return

    person_centroids = {}

    for person_name in os.listdir(IMAGE_DIR):
        person_dir = os.path.join(IMAGE_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"Processing images for {person_name}...")
        embeddings = []
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                # Use robust Facenet model from DeepFace
                result = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
                if len(result) > 0:
                    emb = result[0]["embedding"]
                    # Normalize the embedding (L2 normalization is best practice for Facenet)
                    emb = emb / np.linalg.norm(emb)
                    embeddings.append(emb)
            except Exception as e:
                print(f"Could not process {img_path}: {e}")

        if embeddings:
            # Calculate the centroid (mean embedding) for the person
            centroid = np.mean(embeddings, axis=0)
            # Normalize centroid again
            centroid = centroid / np.linalg.norm(centroid)
            person_centroids[person_name] = centroid
            print(f"Computed FaceNet centroid for {person_name} from {len(embeddings)} images.")
        else:
            print(f"No valid images found for {person_name}.")

    # Save to file
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(person_centroids, f)
    print(f"Training complete. Embeddings saved to '{EMBEDDINGS_FILE}'.")

if __name__ == "__main__":
    train_embeddings()
