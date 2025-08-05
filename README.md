## Classify the handwriting of EMNIST
### Information
- The EMNIST dataset is an extension of the MNIST dataset, which includes more handwritten letters and digits. This project focuses on classifying letters from the EMNIST using a CNN model.
- The datasets are downloaded from Kaggle: https://www.kaggle.com/datasets/crawford/emnist/data. The main dataset is the letters EMINST subset, which consists of 103600 characters with 26 classes. Each image is 28x28 pixels, grayscale.
- The CNN model used in this project included the following layers: convolutional layers with ReLU activation, max-pooling layers, fully connected (dense) layers, and dropout layers for regularization. The model is trained using the Adam optimizer and categorical cross-entropy loss.
- The model achieves high accuracy in classifying handwritten letters from the EMNIST dataset. Below are some sample results: Training Accuracy: ~98% and Validation Accuracy: ~92%. More detailed results and visualizations can be found in the notebook.

### Deploy with Flask
https://github.com/user-attachments/assets/04051eeb-01fd-4b15-8ab2-71b2a9e33fcf
