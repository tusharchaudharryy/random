import os
import yaml
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from audio_classifier.logger import logger 

def process_and_save_audio(config_path="params.yaml"):
    """
    Loads raw audio files, processes them to a fixed length,
    and saves them to the processed data directory.
    """
    logger.info("Starting audio preprocessing pipeline...")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    sample_rate = config["preprocessing"]["sample_rate"]
    duration = config["preprocessing"]["duration_seconds"]
    
    target_length = sample_rate * duration

    os.makedirs(processed_dir, exist_ok=True)
    logger.info(f"Ensured processed data directory exists at: {processed_dir}")

    for class_folder in os.listdir(raw_dir):
        class_path = os.path.join(raw_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        logger.info(f"Processing class: {class_folder}")
        
        output_class_path = os.path.join(processed_dir, class_folder)
        os.makedirs(output_class_path, exist_ok=True)
        

        for filename in tqdm(os.listdir(class_path), desc=f"  -> {class_folder} files"):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_path, filename)
                
                try:
                    signal, sr = librosa.load(file_path, sr=sample_rate)

                    if len(signal) > target_length:
                        signal = signal[:target_length]
                    else:
                        signal = np.pad(signal, (0, target_length - len(signal)), "constant")
                    
                    output_filepath = os.path.join(output_class_path, filename)
                    sf.write(output_filepath, signal, sample_rate)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

    logger.info("Audio preprocessing complete!")


if __name__ == '__main__':
    process_and_save_audio()