import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset_path = "dataset"
model_path = "model.keras"

def create_model():
    model = Sequential([
        Input(shape=(64, 64, 3)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(26, activation="softmax")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def get_image_paths_and_labels(dataset_path):
    image_paths = []
    labels = []
    
    for label in range(26):
        char_label = chr(label + 65)
            
        folder_path = os.path.join(dataset_path, char_label)
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".jpg"):
                    image_paths.append(os.path.join(folder_path, file_name))
                    labels.append(label)
                    
    return image_paths, labels

def load_images(image_paths, labels):
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        images.append(img)
        
    return np.array(images), np.array(labels)

def create_data_generator():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest"
    )


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(history.history["accuracy"], label="Training Accuracy", marker="o", color="blue")
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="o", color="cyan")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid()

    ax2.plot(history.history["loss"], label="Training Loss", marker="x", color="red")
    ax2.plot(history.history["val_loss"], label="Validation Loss", marker="x", color="orange")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()

def main():
    model = create_model()
    
    image_paths, labels = get_image_paths_and_labels(dataset_path)
    x_data, y_data = load_images(image_paths, labels)
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )
    
    datagen = create_data_generator()
    datagen.fit(x_train)
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        validation_data=(x_val/255.0, y_val),
        epochs=100,
        callbacks=[early_stopping]
    )
    
    model.save(model_path)
    
    plot_training_history(history)

if __name__ == "__main__":
    main()