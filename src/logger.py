import logging
from datetime import datetime
import os

log_folder = os.path.join(os.getcwd(), "Logs")
os.makedirs(log_folder, exist_ok=True)

log_filename = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')}.log" 
log_file_path = os.path.join(log_folder, log_filename)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s",
    level=logging.DEBUG,
    filemode="a"
)

# Test logging
#logging.info("Logging setup complete.")
#logging.debug("This is a debug message.")
