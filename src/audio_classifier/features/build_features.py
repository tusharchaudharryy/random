import os
import yaml
import librosa
import numpy as np
from tqdm import tqdm
from audio_classifier.logger import logger 

def extract_features(config_path="params.yaml"):
    """
    Loads processed audio, extracts Mel-spectrograms, and saves
    them to the features directory.
    """
    logger.info("Starting feature extraction pipeline...")


    with open(config_path) as f:
        config = yaml.safe_load(f)

    processed_dir = config["data"]["processed_dir"]
    features_path = config["data"]["features_path"]
    sample_rate = config["preprocessing"]["sample_rate"]
    n_mels = config["features"]["n_mels"]
    n_fft = config["features"]["n_fft"]
    hop_length = config["features"]["hop_length"]
    
    features = []
    labels = []
    class_map = {"clean": 0, "infested": 1}

    os.makedirs(features_path, exist_ok=True)
    logger.info(f"Ensured features directory exists at: {features_path}")


    for class_name, label in class_map.items():
        class_dir = os.path.join(processed_dir, class_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"Directory not found for class '{class_name}'. Skipping.")
            continue
            
        logger.info(f"Extracting features for class: {class_name}")

        for filename in tqdm(os.listdir(class_dir), desc=f"  -> {class_name} files"):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                
                try:
                    signal, sr = librosa.load(file_path, sr=sample_rate)
                    
                    mel_spectrogram = librosa.feature.melspectrogram(
                        y=signal, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
                    )
                    
                    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
                    
                    features.append(log_mel_spectrogram)
                    labels.append(label)

                except Exception as e:
                    logger.error(f"Error extracting features from {file_path}: {e}")
    
    features_array = np.array(features)
    labels_array = np.array(labels)

    np.save(os.path.join(features_path, "features.npy"), features_array)
    np.save(os.path.join(features_path, "labels.npy"), labels_array)

    logger.info("Feature extraction complete!")
    logger.info(f"Features shape: {features_array.shape}")
    logger.info(f"Labels shape: {labels_array.shape}")

if __name__ == '__main__':
    extract_features()