import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import datetime

def train_model(data_path, log_dir="../logs"):
    # Load preprocessed data
    data = np.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    # Define the model
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model
    x = Flatten()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)  # Binary classification (damaged/undamaged)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Define callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint("../models/car_damage_model.h5", save_best_only=True, monitor="val_loss")

    # Set up TensorBoard callback
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=f"{log_dir}/train_{current_time}", histogram_freq=1)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint, tensorboard_callback]
    )

    print("Model trained and saved to ../models/car_damage_model.h5")
    return history

if __name__ == "__main__":
    data_path = "../data/preprocessed_data.npz"
    train_model(data_path)