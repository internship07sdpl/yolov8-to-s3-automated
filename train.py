from roboflow import Roboflow
from ultralytics import YOLO
import boto3
import os
import sys

print("Starting YOLOv8 training pipeline...")

# ------------------------------
# Download dataset
# ------------------------------
try:
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("perception-models").project("tata_ace_exterior")
    version = project.version(2)

    print("Downloading dataset...")
    dataset = version.download("yolov8")

except Exception as e:
    print(f"Dataset error: {e}")
    sys.exit(1)

# -------------------------------
# Load model
# -------------------------------
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

# -------------------------------
# Train model
# -------------------------------
print("Training model...")

try:
    model.train(
        data=dataset.location + "/data.yaml",
        epochs=2,
        imgsz=320,
        batch=2,
        workers=0,
        project="runs",
        name="train"
    )
except Exception as e:
    print(f"Training failed: {e}")
    sys.exit(1)

# -------------------------------
# Get model path
# -------------------------------
best_model_path = str(model.trainer.save_dir / "weights/best.pt")
print("Model saved at:", best_model_path)

if not os.path.exists(best_model_path):
    print("ERROR: best.pt not found!")
    sys.exit(1)

# -------------------------------
# Save path for GitHub Actions
# -------------------------------
with open("model_path.txt", "w") as f:
    f.write(best_model_path)

print("Saved model path to model_path.txt")

# -------------------------------
# Upload to S3
# -------------------------------
try:
    print("Uploading model to S3...")
    s3 = boto3.client('s3')

    s3.upload_file(
        best_model_path,
        "best2.pt",
        "models/best.pt"
    )

    print("Uploaded to S3 successfully!")

except Exception as e:
    print(f"S3 upload failed: {e}")

print("Pipeline completed successfully!")
