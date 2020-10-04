# Robocup Metrics Experiments

## Neural Networks

The following neural networks have architectures that were selected empirically with the purpose of validating our proposed metrics.

#### MLP (Multi-Layer Perceptron) Network

This neural network consists of 2 fully connected hidden layers of size 500 each with the ReLU activation function applied after each hidden layer.
We also apply batch normalization on the input vector and after each hidden layer to improve the convergence rate of the network's parameters.
The neural network layers are defined by the configuration 9-500-500-5.

#### LSTM (Long Short-Term Memory) Network

The LSTM neural network consists of a single LSTM cell that has a hidden dimension of 128.
The ReLU activation function is applied to the output of the LSTM cell, followed by a fully connected layer that outputs the final action it selected (i.e. a fully connected layer with the configuration 128-5).
Batch normalization is applied on the input vector as well as between the ReLU activation function and the fully connected layer.

## Hyper-parameters

The same hyper-parameters and training method are used for both Neural Networks. The hyper-parameters were selected empirically:

#### DAgger (Data Aggregation) Hyper-parameters
- β = 0
- t-step = 10
- Epochs at the end of each t-step trajectory = 10

#### Neural Network Hyper-parameters
Adaptive learning rate optimization (Adam):
- Learning rate = 0.0001
- Epochs after termination of DAgger = 200

Loss: cross-entropy loss

## Dataset

All dataset sizes were selected qualitatively to make the MLP network outperform the LSTM network in imitating the reactive expert, and to make the LSTM network outperform the MLP network in imitating the state-based expert.

#### MLP
- 56,041 time steps for the reactive (krislet) expert.
- 19,750 time steps for the state-based expert.

#### LSTM
- 59,658 time steps for the reactive (krislet) expert.
- 26,153 time steps for the state-based expert.

## Training Method (DAgger)

At every step of a run, the agent selects an action (using the neural network) based on the current perception of the environment and performs the action they selected.
Next, it queries the expert for the action it would have taken at that step of the run and add the environment action pair to the training set of the agent.
The agent runs 10 epochs of training on the new dataset at the end of each 10-step trajectory using the cross-entropy loss criterion and adaptive learning rate optimization (Adam).