from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import base64
import io
import traceback

app = Flask(__name__)
CORS(app)

# Global variables for model and processor
MODEL_PATH = r"C:\Users\tcong\OneDrive\Desktop\models\detr-resnet-50"
processor = None
model = None

def load_model():
    global processor, model
    try:
        print("Loading model...")
        processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
        model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "Server is running!"})

@app.route('/detect', methods=['POST'])
def detect_objects():
    if not processor or not model:
        return jsonify({
            "success": False,
            "error": "Model not loaded"
        }), 500

    try:
        print("Receiving detection request...")
        
        # Get the image from the request
        if not request.json or 'image' not in request.json:
            return jsonify({
                "success": False,
                "error": "No image data received"
            }), 400

        image_data = request.json['image']
        
        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Convert base64 to image
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            print("Image successfully decoded")
        except Exception as e:
            print(f"Error decoding image: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Invalid image data"
            }), 400

        # Process image with model
        print("Processing image with model...")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Post-process outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=0.7
        )[0]

        # Format detections
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.7:  # Confidence threshold
                box_coords = box.tolist()
                detections.append({
                    "label": model.config.id2label[label.item()],
                    "confidence": float(score.item()),
                    "box": {
                        "x": float(box_coords[0]),
                        "y": float(box_coords[1]),
                        "width": float(box_coords[2] - box_coords[0]),
                        "height": float(box_coords[3] - box_coords[1])
                    }
                })

        print(f"Detection complete. Found {len(detections)} objects.")
        return jsonify({
            "success": True,
            "detections": detections
        })
        
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    if load_model():
        print("Starting server on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load model. Server not started.")