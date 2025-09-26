import os

# Project directory name
PROJECT_NAME = "audio-classification-tinyml"

# File & folder structure
dirs = [
    f"{PROJECT_NAME}/.dvc/",
    f"{PROJECT_NAME}/data/raw/",
    f"{PROJECT_NAME}/data/features/",
    f"{PROJECT_NAME}/models/",
    f"{PROJECT_NAME}/notebooks/",
    f"{PROJECT_NAME}/src/audio_classifier/data/",
    f"{PROJECT_NAME}/src/audio_classifier/features/",
    f"{PROJECT_NAME}/src/audio_classifier/models/",
    f"{PROJECT_NAME}/src/audio_classifier/deployment/",
    f"{PROJECT_NAME}/tests/"
]

files = [
    f"{PROJECT_NAME}/src/audio_classifier/__init__.py",
    f"{PROJECT_NAME}/src/audio_classifier/data/make_dataset.py",
    f"{PROJECT_NAME}/src/audio_classifier/features/build_features.py",
    f"{PROJECT_NAME}/src/audio_classifier/models/architecture.py",
    f"{PROJECT_NAME}/src/audio_classifier/models/train_model.py",
    f"{PROJECT_NAME}/src/audio_classifier/models/evaluate_model.py",
    f"{PROJECT_NAME}/src/audio_classifier/deployment/convert_to_tflite.py",
    f"{PROJECT_NAME}/.gitignore",
    f"{PROJECT_NAME}/dvc.yaml",
    f"{PROJECT_NAME}/params.yaml",
    f"{PROJECT_NAME}/requirements.txt",
    f"{PROJECT_NAME}/README.md"
]

def create_project_structure():
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    for file_path in files:
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("")  # Empty file
            print(f"Created empty file: {file_path}")
        else:
            print(f"File already exists: {file_path}")

if __name__ == "__main__":
    create_project_structure()
    print("\n✅ Project structure created successfully!")
