from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import os
import subprocess

# Define script paths
SCRIPTS_DIR = os.path.join(os.getcwd(), "scripts")
DATA_PATH = os.path.join(os.getcwd(), "data/preprocessed_data.npz")
MODEL_PATH = os.path.join(os.getcwd(), "models/car_damage_model.h5")

# Define Python functions for each step
def preprocess_data():
    subprocess.run(["python", f"{SCRIPTS_DIR}/preprocess.py"], check=True)

def train_model():
    subprocess.run(["python", f"{SCRIPTS_DIR}/train_model.py"], check=True)

def evaluate_model():
    subprocess.run(["python", f"{SCRIPTS_DIR}/evaluate_model.py"], check=True)

# Define the Airflow DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG(
    'ml_pipeline_dag',
    default_args=default_args,
    schedule_interval=None,  # Run on demand
) as dag:
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model
    )

    # Define task dependencies
    preprocess_task >> train_task >> evaluate_task