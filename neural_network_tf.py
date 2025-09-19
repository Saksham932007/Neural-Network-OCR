import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def load_and_prepare_data(train_path, test_path):
    """
    Loads and preprocesses the MNIST data from CSV files.
    For a CNN, we need to reshape the data into a 2D image format.
    """
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate labels and features, and normalize pixel values
    y_train = train_df['label'].values
    x_train = train_df.drop('label', axis=1).values / 255.0
    
    y_test = test_df['label'].values
    x_test = test_df.drop('label', axis=1).values / 255.0

    # --- Reshape for CNN ---
    # The CNN expects a 4D tensor: (num_samples, height, width, channels)
    # MNIST images are 28x28 pixels with 1 color channel (grayscale).
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    print("Data loaded and preprocessed for CNN.")
    return x_train, y_train, x_test, y_test

def build_cnn_model():
    """
    Builds an improved Convolutional Neural Network (CNN) to achieve >99% accuracy.
    This version includes Batch Normalization and more layers.
    """
    model = tf.keras.models.Sequential([
        # Input layer is now implicitly handled by the data augmentation layer
        
        # --- Convolutional Base ---
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # --- Classifier Head ---
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Using a learning rate scheduler can help fine-tune the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main function to run the TensorFlow/Keras CNN."""
    # --- Configuration ---
    EPOCHS = 20 # More epochs to learn from augmented data
    BATCH_SIZE = 128
    
    # --- Data Loading ---
    x_train, y_train, x_test, y_test = load_and_prepare_data('mnist_train.csv', 'mnist_test.csv')

    # --- Data Augmentation ---
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,     # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
    )
    datagen.fit(x_train)

    # --- Model Building & Training ---
    print("\nBuilding and training Advanced Convolutional Neural Network (CNN)...")
    model = build_cnn_model()
    
    model.summary()
    
    start_time = time.time()
    
    # Use model.fit with the data generator
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        steps_per_epoch=x_train.shape[0] // BATCH_SIZE # Number of steps per epoch
    )
    
    duration = time.time() - start_time
    print(f"\nTraining finished in {duration:.2f} seconds.")
    
    # --- Evaluation ---
    print("\nEvaluating final model on test data...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    # --- Visualization ---
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.1)
    plt.title("Model Training History")
    plt.xlabel("Epoch")
    plt.show()

if __name__ == "__main__":
    main()


