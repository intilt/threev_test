import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import os
import datetime

def evaluate_model(data_path, model_path, log_dir="../logs"):
    # Load preprocessed data
    data = np.load(data_path)
    X_val, y_val = data["X_val"], data["y_val"]

    # Load the trained model
    model = load_model(model_path)

    # Make predictions
    y_pred_probs = model.predict(X_val)
    y_pred = (y_pred_probs > 0.5).astype("int32")

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="binary")
    recall = recall_score(y_val, y_pred, average="binary")
    f1 = f1_score(y_val, y_pred, average="binary")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["Damaged", "Undamaged"]))

    # Log evaluation metrics to TensorBoard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_log_dir = os.path.join(log_dir, f"evaluation_{current_time}")
    os.makedirs(eval_log_dir, exist_ok=True)

    with tf.summary.create_file_writer(eval_log_dir).as_default():
        tf.summary.scalar("Accuracy", accuracy, step=0)
        tf.summary.scalar("Precision", precision, step=0)
        tf.summary.scalar("Recall", recall, step=0)
        tf.summary.scalar("F1 Score", f1, step=0)

    print(f"Evaluation metrics logged to TensorBoard in {eval_log_dir}")

if __name__ == "__main__":
    data_path = "../data/preprocessed_data.npz"
    model_path = "../models/car_damage_model.h5"
    evaluate_model(data_path, model_path)