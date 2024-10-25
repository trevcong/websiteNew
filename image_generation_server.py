from flask import Flask, request, jsonify
from flask_cors import CORS
from diffusers import AutoPipelineForText2Image
import torch
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

# Initialize the model
print("Loading FLUX model...")
try:
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "blackforest-labs/FLUX", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    pipeline = None

@app.route('/generate', methods=['POST'])
def generate_image():
    if not pipeline:
        return jsonify({
            "success": False,
            "error": "Model not loaded"
        }), 500

    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({
                "success": False,
                "error": "No prompt provided"
            }), 400

        print(f"Generating image for prompt: {prompt}")

        # Generate image
        image = pipeline(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]

        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{img_str}"
        })

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Using different port than object detection