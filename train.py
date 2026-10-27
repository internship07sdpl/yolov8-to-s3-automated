from roboflow import Roboflow
from ultralytics import YOLO
import os
import sys
import shutil
import boto3
import json

print("Starting YOLOv8 SageMaker training pipeline...")

# Retrieve Roboflow API key securely
try:
    secretsmanager = boto3.client(
        "secretsmanager",
        region_name="ap-south-1"
    )

    response = secretsmanager.get_secret_value(
        SecretId="roboflow/training-api-key"
    )

    secret_data = json.loads(response["SecretString"])
    roboflow_api_key = secret_data["ROBOFLOW_API_KEY"]

    print("Roboflow credential retrieved successfully")

except Exception as e:
    print("Credential retrieval failed:", type(e).__name__)
    sys.exit(1)

# --------------------------------------------------
# Download Dataset from Roboflow
# --------------------------------------------------

try:
    rf = Roboflow(api_key=roboflow_api_key)

    project = rf.workspace("perception-models").project(
        "tata_ace_exterior"
    )

    version = project.version(2)

    print("Downloading dataset...")

    dataset = version.download("yolov8")

except Exception as e:
    print(f"Dataset error: {e}")
    sys.exit(1)

# --------------------------------------------------
# Load YOLO Model
# --------------------------------------------------

print("Loading YOLOv8 model...")

model = YOLO("yolov8n.pt")

# --------------------------------------------------
# Train Model
# --------------------------------------------------

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

# --------------------------------------------------
# Locate Training Output Folder
# --------------------------------------------------

training_output_dir = str(model.trainer.save_dir)

print("Training output directory:")
print(training_output_dir)

if not os.path.exists(training_output_dir):
    print("ERROR: Training output directory not found!")
    sys.exit(1)

# --------------------------------------------------
# SageMaker Output Directory
# --------------------------------------------------

output_dir = "/opt/ml/model"

os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# Copy Complete YOLO Training Outputs
# --------------------------------------------------

destination_dir = f"{output_dir}/training_outputs"

if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)

shutil.copytree(
    training_output_dir,
    destination_dir
)

print("All YOLO training outputs copied successfully")

# --------------------------------------------------
# Copy best.pt Separately
# --------------------------------------------------

best_model_path = str(
    model.trainer.save_dir / "weights/best.pt"
)

if os.path.exists(best_model_path):
    shutil.copy(
        best_model_path,
        f"{output_dir}/best.pt"
    )

    print("best.pt copied successfully")

else:
    print("WARNING: best.pt not found")

# --------------------------------------------------
# Finished
# --------------------------------------------------

print("Training pipeline completed successfully!")
