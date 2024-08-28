# TensorFlow Playground Activity

> 🚦 **Instructions:** Fill out this quiz by marking the correct option(s). In order to save your answers, use your browser's `Print to PDF` function. ([Chrome instructions.](https://libguides.rowan.edu/c.php?g=248114&p=4710174))

TensorFlow Playground is an interactive tool that allows you to visualize and experiment with neural networks in your browser. It’s a great way to understand the impact of different hyperparameters, network architectures, and data patterns on model performance. In this activity, you will explore key concepts related to neural networks using TensorFlow Playground.

---

## **Part 1: Important Concepts**

**1. Neural Networks:**

- **Definition:** A computational model inspired by the human brain, consisting of layers of nodes (neurons) that process data and learn from it.
- **Components:** 
    - **Input Layer:** Receives the input data.
    - **Hidden Layers:** Perform computations and feature transformations.
    - **Output Layer:** Produces the final prediction or classification.

**2. Activation Functions:**

- **Definition:** Functions that determine whether a neuron should be activated or not. They introduce non-linearity into the model.
- **Common Types:**
    - **ReLU (Rectified Linear Unit):** $f(x) = max(0, x)$
    - **Sigmoid:** $f(x) = 1 / (1 + e^-x)$
    - **Tanh:** $f(x) = (e^x - e^-x) / (e^x + e^-x)$

**3. Learning Rate:**

- **Definition:** A hyperparameter that controls how much to adjust the model's weights with respect to the loss gradient.
- **Impact:** Too high can cause divergence, too low can make the training process slow.

**4. Regularization:**
   
- **Definition:** Techniques used to prevent overfitting by penalizing large weights.
- **Types:**
    - **L1 Regularization:** Adds a penalty equal to the absolute value of the magnitude of coefficients.
    - **L2 Regularization:** Adds a penalty equal to the square of the magnitude of coefficients.

**5. Overfitting vs. Underfitting:**

- **Overfitting:** When the model performs well on training data but poorly on new, unseen data.
- **Underfitting:** When the model is too simple to capture the underlying pattern of the data.

**6. Epochs:**

- **Definition:** One complete pass of the training dataset through the neural network.
- **Impact:** More epochs allow the model to learn better but can also lead to overfitting.

**7. Data Patterns:**

- **Understanding:** The shape and distribution of your data can significantly affect the neural network's ability to learn. Common patterns include linear, spiral, circular, and XOR.

**8. Weight Initialization:**

- **Definition:** The initial values assigned to the weights in the neural network.
- **Impact:** Proper weight initialization can lead to faster convergence and better overall performance.

**9. Batch Size:**
   
- **Definition:** The number of training examples used to calculate the gradient at each iteration.
- **Impact:** A smaller batch size can lead to noisier updates but may generalize better, while a larger batch size provides a smoother gradient but might require more epochs.

---

## **Part 2: Exercises**

> 💡 **Tip:** You can save your TensorFlow Playground experiments by copying the URL from your browser's address bar. The URL contains all the settings of your current experiment, including the network architecture, activation functions, data set, learning rate, and other parameters. You can save it, or share it with others to revisit or collaborate on the same experiment later.

### **Exercise 1: Understanding Activation Functions**

1. **Objective:** Experiment with different activation functions and observe their impact on model performance.
2. **Steps:**
   - Open [TensorFlow Playground](https://playground.tensorflow.org/).
   - Use a dataset with a clear pattern (e.g., "Circle" or "XOR").
   - Try different activation functions ($ReLU$, $Sigmoid$, $Tanh$) in the hidden layers.
   - Observe how the decision boundary changes and how well the model fits the data.
3. **Questions:**
   - How does the decision boundary change with different activation functions?
   - Which activation function provided the best fit for the dataset?

### **Exercise 2: Exploring Learning Rates**

1. **Objective:** Understand the importance of learning rates in training neural networks.
2. **Steps:**
   - Select a complex dataset (e.g., "Spiral").
   - Set up a neural network with 2 hidden layers and several neurons.
   - Experiment with different learning rates (e.g., `0.01`, `0.1`, `0.3`).
   - Observe how the model's convergence speed and accuracy are affected.
3. **Questions:**
   - What happened when the learning rate was too high or too low?
   - How did the learning rate affect the model's ability to generalize to unseen data?

### **Exercise 3: Regularization Techniques**

1. **Objective:** Explore how regularization affects model performance and prevents overfitting.
2. **Steps:**
   - Choose a dataset prone to overfitting (e.g., "High noise").
   - Set up a neural network with multiple hidden layers.
   - Experiment with different regularization settings (`L1`, `L2`) and observe their impact.
   - Compare results with and without regularization.
3. **Questions:**
   - How did regularization affect the model's complexity and performance?
   - Which regularization technique was most effective in preventing overfitting?

### **Exercise 4: Impact of Epochs on Training**

1. **Objective:** Observe how the number of epochs influences model performance.
2. **Steps:**
   - Select a simple dataset (e.g., "Circle").
   - Set up a basic neural network with 1 or 2 hidden layers.
   - Train the model with varying numbers of epochs (e.g., `10`, `50`, `100`).
   - Observe the model’s performance on both training and test sets.
3. **Questions:**
   - What changes occurred in the decision boundary as the number of epochs increased?
   - Did you notice any signs of overfitting with more epochs?

### **Exercise 5: Experimenting with Weight Initialization**

1. **Objective:** Understand the importance of weight initialization on the convergence of a neural network.
2. **Steps:**
   - Select a dataset with moderate complexity (e.g., "Gaussian").
   - Set up a neural network with 2 hidden layers.
   - Experiment with different weight initialization methods (e.g., small random values, zeros).
   - Observe the training process and convergence speed.
3. **Questions:**
   - How did different initialization methods affect the learning process?
   - Which initialization method resulted in faster or more stable convergence?

### **Exercise 6: Analyzing the Effect of Batch Size**

1. **Objective:** Explore how different batch sizes affect model training and generalization.
2. **Steps:**
   - Select a dataset with a non-linear pattern (e.g., "Spiral").
   - Set up a neural network with 3 hidden layers.
   - Train the model using different batch sizes (e.g., `1`, `10`, `100`).
   - Compare the performance, speed of convergence, and generalization ability of the model.
3. **Questions:**
   - How did the model’s performance vary with different batch sizes?
   - Did smaller or larger batch sizes lead to better generalization?

---

## **Part 3: Quiz**

**1.** What is the role of the activation function in a neural network?

- [ ] To determine the learning rate
- [ ] To decide whether a neuron should be activated
- [ ] To control the number of layers in the network
- [ ] To measure the error rate during training

**2.** How does regularization help prevent overfitting?

- [ ] By increasing the number of epochs
- [ ] By adding noise to the input data
- [ ] By penalizing large weights in the model
- [ ] By reducing the learning rate

**3.** What is a potential consequence of setting the learning rate too high?

- [ ] The model will converge too quickly
- [ ] The model will learn the data perfectly
- [ ] The model might diverge and fail to learn
- [ ] The model will underfit the data

**4.** How many epochs should you train a model for to avoid overfitting?

- [ ] As many as possible
- [ ] Enough to achieve low training error, but not too many that test error increases
- [ ] Only one epoch
- [ ] Until the model reaches 100% accuracy on training data

**5.** What impact does a larger batch size have on training a neural network?

- [ ] It increases the noise in gradient updates
- [ ] It leads to smoother gradient updates
- [ ] It requires fewer epochs to converge
- [ ] It always improves model performance

**6.** Why is proper weight initialization important in neural networks?

- [ ] To avoid vanishing gradients
- [ ] To ensure that all neurons in a layer are identical
- [ ] To speed up the training process
- [ ] To reduce the number of epochs required

**7.** Which data pattern would be most challenging for a simple neural network to learn?

- [ ] Linear
- [ ] Circular
- [ ] Spiral
- [ ] Gaussian

**8.** What happens if you do not use an activation function in the hidden layers of a neural network?

- [ ] The network becomes a linear model
- [ ] The network overfits more easily
- [ ] The network cannot converge
- [ ] The network requires fewer epochs to train

**9.** Which of the following is a symptom of underfitting?

- [ ] High accuracy on training data but low accuracy on test data
- [ ] Low accuracy on both training and test data
- [ ] High accuracy on both training and test data
- [ ] Model performs well on complex patterns but not on simple ones

**10.** When experimenting with TensorFlow Playground, which hyperparameter has the most direct impact on the speed of convergence?

- [ ] Number of layers
- [ ] Regularization type
- [ ] Learning rate
- [ ] Batch size
