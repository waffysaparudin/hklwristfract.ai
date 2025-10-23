from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import base64
from PIL import Image
import io

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "trained_yolov8n.pt")
model = YOLO(model_path)

@app.route("/", methods=["GET"])
def home():
    return {"status": "Fract.AI API is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = "temp.jpg"
    file.save(file_path)

    results = model.predict(source=file_path, conf=0.25)
    preds = results[0].tojson()

    # Save annotated image
    annotated_img_path = "annotated.jpg"
    results[0].save(filename=annotated_img_path)

    # Convert image to Base64
    with open(annotated_img_path, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

    return jsonify({
        "predictions": preds,
        "annotated_image_base64": encoded_img
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
