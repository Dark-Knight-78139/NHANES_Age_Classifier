import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_step(script_name):
    logger.info(f"Running {script_name}...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Error running {script_name}: {result.stderr}")
        return False
    logger.info(result.stdout)
    return True

def main():
    steps = [
        "src/data_ingestion.py",
        "src/preprocessing.py",
        "src/model_training.py"
    ]
    
    for step in steps:
        if not run_step(step):
            logger.error("Pipeline failed.")
            return

    logger.info("Full pipeline completed successfully!")

if __name__ == "__main__":
    main()
