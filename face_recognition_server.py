from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io
import base64
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize face detection and recognition models
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Load the FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval()
if torch.cuda.is_available():
    resnet = resnet.cuda()

# Directory for storing user face embeddings
EMBEDDINGS_DIR = 'face_embeddings'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def get_face_embedding(face_img):
    """Convert face image to embedding vector"""
    # Convert to RGB if needed
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    elif face_img.shape[2] == 4:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)
    
    # Convert to PIL Image
    face_img = Image.fromarray(face_img)
    
    # Get face embedding
    face_tensor = mtcnn(face_img)
    if face_tensor is None:
        return None
        
    if torch.cuda.is_available():
        face_tensor = face_tensor.cuda()
    
    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0))
    
    return embedding.cpu().numpy()[0]

def compare_embeddings(embedding1, embedding2, threshold=0.7):
    """Compare two face embeddings"""
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity > threshold, float(similarity)

@app.route('/api/train', methods=['POST'])
def train_face():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        image_data = data.get('image')
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        
        # Get face embedding
        embedding = get_face_embedding(image)
        if embedding is None:
            return jsonify({'error': 'No face detected in image'}), 400
            
        # Save embedding
        embedding_file = os.path.join(EMBEDDINGS_DIR, f'{user_id}.npy')
        np.save(embedding_file, embedding)
        
        # Save metadata
        metadata_file = os.path.join(EMBEDDINGS_DIR, f'{user_id}_metadata.json')
        metadata = {
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'num_training_images': 1
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify', methods=['POST'])
def verify_face():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        image_data = data.get('image')
        
        # Load stored embedding
        embedding_file = os.path.join(EMBEDDINGS_DIR, f'{user_id}.npy')
        if not os.path.exists(embedding_file):
            return jsonify({'error': 'No trained face data found'}), 404
        stored_embedding = np.load(embedding_file)
        
        # Decode and process new image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        
        # Get face embedding
        new_embedding = get_face_embedding(image)
        if new_embedding is None:
            return jsonify({'error': 'No face detected in image'}), 400
            
        # Compare embeddings
        is_match, confidence = compare_embeddings(stored_embedding, new_embedding)
        
        return jsonify({
            'match': bool(is_match),
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)