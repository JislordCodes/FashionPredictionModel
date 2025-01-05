# Fashion MNIST Classification Model

This project demonstrates a simple neural network model implemented using TensorFlow and Keras to predicts with aboout an 80% accuracy the name of the images of differnt fashion items. 
---

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Author](#author)

---

## Dataset

The Fashion MNIST dataset consists of 70,000 grayscale images, each of size 28x28 pixels. There are 10 classes of fashion items:

1. T-Shirt/Top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle Boot

- **Training Data**: 60,000 images
- **Testing Data**: 10,000 images

---

## Model Architecture

The model is a simple feedforward neural network with the following layers:
1. **Flatten Layer**: Converts the 28x28 images into a 1D array of 784 features.
2. **Dense Layer**: Fully connected layer with 128 neurons and ReLU activation function.
3. **Output Layer**: Fully connected layer with 10 neurons (one for each class) and softmax activation for probability distribution.

---

## Data Preprocessing

1. Normalize the pixel values of the images to a range of 0 to 1 by dividing by 255.
2. Prepare the training and test datasets.

---

## Training

The model was compiled with the following configurations:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

The model was trained for **12 epochs**, achieving a high training accuracy.

---

## Evaluation

The model was evaluated on the test dataset:
- **Test Accuracy**: 87.65%
- **Test Loss**: 0.3487

---

## Usage

1. Load the Model
Clone or copy the repository and run the script to train and evaluate the model.

2. Check Specific Predictions
Use the model to predict the class of a specific image. Example:
python
predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[70])])

3. Visualize Results
Display the test image with the predicted label:
plt.figure()
plt.imshow(test_images[70])
plt.colorbar()
plt.grid(False)
plt.show()
Requirements
Python 3.7 or higher
TensorFlow 2.x
NumPy
Matplotlib

Install dependencies using:
pip install tensorflow numpy matplotlib
