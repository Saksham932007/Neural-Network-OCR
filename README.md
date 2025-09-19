# Modern OCR Neural Network (From Scratch & with TensorFlow)

This project provides two approaches to building an Optical Character Recognition (OCR) neural network for the MNIST dataset. It's an updated version of an earlier from-scratch implementation.

    From-Scratch NumPy Implementation: A modernized, more efficient version of the original code. It uses pure NumPy, is structured in a class, and employs vectorized operations for better performance. This is excellent for understanding the core mechanics of a neural network.

    Professional TensorFlow/Keras Implementation: The standard, modern way to build neural networks. This approach is more concise, powerful, and leverages optimizations like GPU acceleration. This is the recommended approach for production or complex models.

## Project Structure

.
├── README.md # This file
├── requirements.txt # Project dependencies
├── download_mnist.py # Script to download and save the MNIST dataset as CSV
├── neural_network_numpy.py # The updated "from scratch" NumPy implementation
└── neural_network_tf.py # The modern TensorFlow/Keras implementation

Step-by-Step Guide to Run

1. Install Dependencies

First, install all the necessary Python libraries using the requirements.txt file. It's recommended to do this in a virtual environment.

pip install -r requirements.txt

2. Download the Dataset

The original code required a manual MNISTtrain.csv. This script automates the process by downloading the official MNIST dataset and saving it in the required CSV format.

python download_mnist.py

This will create mnist_train.csv and mnist_test.csv in your directory. 3. Run the "From-Scratch" NumPy Version

To see the updated NumPy-based network in action, run the following command. It will train the model and print the accuracy.

python neural_network_numpy.py

4. Run the Professional TensorFlow/Keras Version

To train the much faster and more accurate framework-based model, run this command.

python neural_network_tf.py

You will notice this version trains significantly faster and typically achieves higher accuracy due to the advanced optimizers and architecture.
