# Deep-Learning-CNN-and GAN
Convolution Neural Network and Generative Adversarial Network - Efficient CIFAR-100 Modelling 

This project focuses on two key aspects of deep learning: image classification using Convolutional Neural Networks (CNN) and image generation using Generative Adversarial Networks (GANs). The project specifically explores the CIFAR-100 dataset for both tasks.

## Project Overview

### Part 1: Classification

#### Methodology
- **Approach:** The classification task is performed using a deep Convolutional Neural Network (CNN) with a non-linear activation function ReLU (`φ(x) = max(0, x)`).
- **Dataset:** CIFAR-100 dataset.
- **Model Details:** The network uses hierarchical feature learning, translation invariance, parameter sharing, and pooling layers. The specific model implementation is inspired by the "SmallFiltersCNN" approach, which is a streamlined version suited for environments where a lighter model with fewer parameters is desired.
- **Mathematical Formulation:**

![Classification mathmetical notation part1](image1.png)</br>
![Classification mathmetical notation part2](image2.png)</br>

#### Results
- **Parameters:** The network has 57,332 parameters.
- **Performance:**
  - Training Accuracy: 41.2%
  - Testing Accuracy: 33.9%
  - Optimization Steps: 10,000
  - Training Loss: 2.338
- **Conclusion:** The results indicate moderate success in classifying the CIFAR-100 dataset. Further tuning and exploration of different architectures or activation functions could potentially improve performance.

#### Limitations
- **Future Improvements:**
  - Use of advanced activation functions like Leaky ReLU or Swish.
  - Incorporation of complex architectures such as residual connections or attention mechanisms.
  - Exploration of data augmentation techniques to enrich the training dataset.

### Part 2: Generative Model

#### Methodology
- **Approach:** A Generative Adversarial Network (GAN) is utilized, specifically a Deep Convolutional GAN (DCGAN), to generate images.
- **Architecture:**
  - **Generator (G):** Transforms random noise vectors \( z \) into synthetic images \( G(z) \).
  - **Discriminator (D):** Distinguishes between real images \( x \) and fake images \( G(z) \).
  - **Objective:** The model is trained using a binary cross-entropy loss function to minimize the difference between real and generated images.

The training objective for the GAN is:

![GAN equation](image3.png)</br>

This adversarial game encourages the generator \( G \) to produce increasingly realistic images, while the discriminator \( D \) learns to better discriminate between real and fake images.

![GAN part 2](image4.png)</br>

#### Results
![image results](image5.png)</br>
Non-Cherry picked samples (left) and interpolants between points in the latent space (right).

- **Parameters:**
  - Generator: 370,624 parameters
  - Discriminator: 167,808 parameters
  - Total: 538,432 parameters
- **Performance:**
  - FID (Fréchet Inception Distance) Score: 66.49
  - Optimization Steps: 50,000
  - The generated images are fairly realistic, though still challenging to distinguish clearly.
- **Conclusion:** The model shows significant improvement during training, suggesting progress in generating images closer to the distribution of real images.

#### Limitations
- **Future Improvements:**
  - Explore alternative generative models or architectures for lower FID scores.
  - Implement techniques like minibatch discrimination to mitigate mode collapse.
  - Use data augmentation to increase the diversity and generalization capability of the model.

## Running the Models

To run the classification and generative models, follow these steps:

1. **Install Required Libraries:** Ensure you have the necessary Python libraries installed, including TensorFlow or PyTorch, depending on the implementation used.
   ```bash
   pip install tensorflow

2. **Run the Classification Model:**
  ```
  python classification_model.py
  ```
3. **Run the Generative Model:**
  ```
  python generative_model.py
  ```
