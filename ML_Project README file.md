# Animal Image Classification using CNN

This project is a Convolutional Neural Network (CNN)-based image classifier developed using TensorFlow and Keras. It classifies animal images into four categories: **Cat**, **Dog**, **Lion**, and **Elephant**.

## 📁 Project Structure

```
ML_PROJECT/
├── ML_PROJECT.ipynb        # Jupyter Notebook with full implementation
├── dataset/                # Dataset folder structured as:
│   ├── train/
│   │   ├── Cat/
│   │   ├── Dog/
│   │   ├── Lion/
│   │   └── Elephant/
│   └── test/
│       ├── Cat/
│       ├── Dog/
│       ├── Lion/
│       └── Elephant/
```

## 🚀 Features

- Image normalization and augmentation using `ImageDataGenerator`
- CNN with three convolutional layers and max pooling
- Dense and dropout layers for classification
- Training and validation accuracy tracking
- Image prediction on new samples

## 🧠 Model Architecture

1. **Input Layer**: Rescaling layer to normalize pixel values.
2. **Conv2D Layer 1**: 16 filters, kernel size 3×3, activation = ReLU → MaxPooling
3. **Conv2D Layer 2**: 32 filters, kernel size 3×3, activation = ReLU → MaxPooling
4. **Conv2D Layer 3**: 64 filters, kernel size 3×3, activation = ReLU → MaxPooling
5. **Flatten Layer**
6. **Dense Layer**: 128 units, activation = ReLU
7. **Dropout Layer**
8. **Output Layer**: Softmax activation for 4 classes

## 🧪 How to Run

1. Clone the repository or open the `ML_PROJECT.ipynb` in Jupyter.
2. Make sure your dataset is structured as shown above.
3. Install the dependencies:
   ```bash
   pip install tensorflow
   ```
4. Run all cells in the notebook to train and test the model.

## 🔍 Sample Prediction

To test on new images:
```python
img = tf.keras.utils.load_img("path_to_image.jpg", target_size=(150,150))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) / 255.0
prediction = model.predict(img_array)
print("Predicted Class:", class_names[np.argmax(prediction)])
```

## 📊 Results

- Achieved good accuracy with minimal overfitting
- Used dropout to enhance generalization

## 📚 Libraries Used

- TensorFlow / Keras
- NumPy
- Matplotlib (optional for plotting)

## 🙋‍♀️ Author

Saloni Navgire  
3rd Year, MGM College of Engineering, Nanded