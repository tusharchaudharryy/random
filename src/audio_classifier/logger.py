import os
import sys
import logging

# Define the logging format
LOGGING_STR = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Define the directory and file path for logs
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Basic configuration for the logger
logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_STR,
    handlers=[
        logging.FileHandler(log_filepath), # Log to a file
        logging.StreamHandler(sys.stdout)  # Log to the console
    ]
)

# Get the logger instance
logger = logging.getLogger("audioClassifierLogger")