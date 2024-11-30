import os
import cv2
import numpy as np

def preprocess_data(data_dir, img_size=(224, 224)):
    def load_dataset(folder_path):
        images, labels = [], []
        for label, folder_name in enumerate(["00-damage", "01-whole"]):
            folder = os.path.join(folder_path, folder_name)
            for file in os.listdir(folder):
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label)
        return np.array(images), np.array(labels)

    train_dir = os.path.join(data_dir, "training")
    val_dir = os.path.join(data_dir, "validation")
    X_train, y_train = load_dataset(train_dir)
    X_val, y_val = load_dataset(val_dir)
    return (X_train / 255.0, y_train), (X_val / 255.0, y_val)

if __name__ == "__main__":
    data_dir = "../data"
    (X_train, y_train), (X_val, y_val) = preprocess_data(data_dir)
    np.savez("../data/preprocessed_data.npz", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print("Preprocessed data saved to ../data/preprocessed_data.npz")