import logging
import os
from datetime import datetime

log = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# ✅ Only folder here
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# ✅ File path
log_file_path = os.path.join(logs_path, log)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    force=True,
    format='[%(asctime)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
