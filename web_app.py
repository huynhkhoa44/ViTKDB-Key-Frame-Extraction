import os
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

# Import TransNetV2
sys.path.append('TransNetV2')
from transnetv2 import TransNetV2

# ----- Initialize Models and Devices -----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
transnet_model = TransNetV2()


# ===== Functions for Key Frame Extraction =====

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_ids = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV default) to RGB for processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_ids.append(i)
    cap.release()
    return frames, frame_ids


def detect_shots(video_path):
    # Detect shot boundaries using TransNetV2 model
    video_frames, single_frame_predictions, all_frame_predictions = transnet_model.predict_video(video_path)
    single_frame_predictions, all_frame_predictions = transnet_model.predict_frames(video_frames)
    shot_boundaries = transnet_model.predictions_to_scenes(single_frame_predictions)
    return shot_boundaries


def extract_features(frames, batch_size=32):
    features = []
    for i in tqdm(range(0, len(frames), batch_size), desc="Extracting features"):
        batch_frames = frames[i:i + batch_size]
        inputs = clip_processor(images=batch_frames, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            batch_features = clip_model.get_image_features(**inputs)
        features.append(batch_features.cpu().numpy())
        torch.cuda.empty_cache()  # Clear GPU memory between batches
    return np.concatenate(features, axis=0)


def estimate_eps(features, k=10):
    if len(features) < k:
        k = len(features)
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(features)
    distances, indices = neighbors_fit.kneighbors(features)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    eps = np.mean(distances)
    return eps


def dbscan_clustering(features, eps, min_samples=4):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(features)
    labels = db.labels_
    return labels


def select_keyframes(frames, frame_ids, features, labels):
    keyframes = []
    keyframe_features = []
    keyframe_indices = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points
        cluster_indices = np.where(labels == label)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_features = features[cluster_indices]
        centroid = cluster_features.mean(axis=0)
        closest_index, _ = pairwise_distances_argmin_min([centroid], cluster_features)
        keyframe_index = cluster_indices[closest_index[0]]
        keyframes.append(frames[keyframe_index])
        keyframe_features.append(features[keyframe_index])
        keyframe_indices.append(frame_ids[keyframe_index])
    return keyframes, keyframe_features, keyframe_indices


def calculate_histogram(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()


def filter_redundant_keyframes(keyframes, keyframe_features, keyframe_indices, similarity_threshold=0.7):
    # Compute color histograms for each keyframe
    histograms = [calculate_histogram(frame) for frame in keyframes]
    informative_keyframes = []
    informative_indices = []
    for i, hist in enumerate(histograms):
        if np.count_nonzero(hist) >= 8:  # simple check for informativeness
            informative_keyframes.append(keyframes[i])
            informative_indices.append(keyframe_indices[i])
    num_keyframes = len(informative_keyframes)
    if num_keyframes == 0:
        return [], []
    # Build similarity matrix based on cosine similarity
    similarity_matrix = np.zeros((num_keyframes, num_keyframes))
    for i in range(num_keyframes):
        for j in range(i + 1, num_keyframes):
            similarity_matrix[i, j] = cosine_similarity([calculate_histogram(informative_keyframes[i])],
                                                        [calculate_histogram(informative_keyframes[j])])[0, 0]
    # Iteratively remove redundant keyframes
    while similarity_matrix.size and np.max(similarity_matrix) >= similarity_threshold:
        i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        informative_keyframes.pop(j)
        informative_indices.pop(j)
        similarity_matrix = np.delete(similarity_matrix, j, axis=0)
        similarity_matrix = np.delete(similarity_matrix, j, axis=1)
    return informative_keyframes, informative_indices


def save_keyframes(keyframes, keyframe_indices, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for frame, frame_index in zip(keyframes, keyframe_indices):
        # Convert from RGB back to BGR before saving
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f'{frame_index}.jpg'), frame_bgr)


def process_video(video_path, output_dir):
    frames, frame_ids = extract_frames(video_path)
    print(f"Extracted {len(frames)} frames from the video.")

    shot_boundaries = detect_shots(video_path)
    print(f"Detected {len(shot_boundaries)} shots in the video.")

    all_keyframes = []
    all_keyframe_indices = []
    for start, end in shot_boundaries:
        shot_frames = frames[start:end + 1]
        shot_frame_ids = frame_ids[start:end + 1]
        features = extract_features(shot_frames)
        if len(features) < 2:
            continue
        eps = estimate_eps(features)
        labels = dbscan_clustering(features, eps)
        keyframes, keyframe_features, keyframe_indices = select_keyframes(shot_frames, shot_frame_ids, features, labels)
        if len(keyframes) == 0:
            continue
        filtered_keyframes, filtered_indices = filter_redundant_keyframes(keyframes, keyframe_features,
                                                                          keyframe_indices)
        all_keyframes.extend(filtered_keyframes)
        all_keyframe_indices.extend(filtered_indices)

    save_keyframes(all_keyframes, all_keyframe_indices, output_dir)
    print(f"Keyframes saved to {output_dir}")
    return output_dir


def main(video_path, output_dir):
    process_video(video_path, output_dir)


# ===== FastAPI Backend =====

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import base64

app = FastAPI()

# Allow cross-origin requests (helpful when calling from Gradio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/extract_keyframes")
async def extract_keyframes_api(file: UploadFile = File(...)):
    # Save the uploaded video temporarily
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create (or clean) a temporary output directory for key frames
    output_dir = "temp_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

    # Process the video
    main(temp_video_path, output_dir)

    # Read saved key frame images and encode them as base64 strings
    keyframes_encoded = []
    # Sort filenames numerically (filenames are like "123.jpg")
    image_files = sorted(os.listdir(output_dir), key=lambda x: int(os.path.splitext(x)[0]))
    for image_file in image_files:
        image_path = os.path.join(output_dir, image_file)
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
            keyframes_encoded.append(encoded_string)

    # Clean up temporary video file (optional)
    os.remove(temp_video_path)

    return {"keyframes": keyframes_encoded}


# ===== Gradio Front End =====

import gradio as gr
import requests
from io import BytesIO
from PIL import Image


def process_video_gradio(video):
    """
    Gradio interface function.
      - Accepts an uploaded video file (the file path is passed by Gradio).
      - Sends it to the FastAPI backend.
      - Receives a JSON response with base64-encoded key frame images.
      - Decodes and returns a list of PIL Image objects.
    """
    # Depending on the Gradio version, 'video' might be a dict or a filepath.
    if isinstance(video, dict) and "name" in video:
        video_path = video["name"]
    else:
        video_path = video

    backend_url = "http://localhost:8000/extract_keyframes"
    with open(video_path, "rb") as f:
        files = {"file": f}
        response = requests.post(backend_url, files=files)
    result = response.json()
    keyframes_base64 = result.get("keyframes", [])
    images = []
    for b64_str in keyframes_base64:
        image_data = base64.b64decode(b64_str)
        image = Image.open(BytesIO(image_data))
        images.append(image)
    return images


# Build the Gradio Interface
iface = gr.Interface(
    fn=process_video_gradio,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Gallery(label="Extracted Key Frames"),
    title="ViTKDB Key Frame Extraction Demo",
    description="Upload a video to extract and display key frames."
)

# ===== Run Both FastAPI and Gradio =====

if __name__ == "__main__":
    import threading
    import uvicorn


    # Function to run FastAPI with uvicorn
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000)


    # Start FastAPI in a background thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    # Launch the Gradio web interface (this will open your browser)
    iface.launch()
