#It provides a built-in way to create log messages and manage log records.
import sys
import logging
import os
from datetime import datetime

from src.exception import CustomException

LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)

os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)
print("LOG_FILE_PATH::::::::",LOG_FILE_PATH)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

# if __name__ == "__main__":
#         logging.info('Logging has started')
        
   
