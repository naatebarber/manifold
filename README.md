# Manifold

Weight separated machine learning framework.

## Network types:
 - `manifold::nn::fc::Manifold` Fully connected feedforward network.

## Trainer types:
 - `manifold::optimizers::MiniBatchGradientDescent` MBGD trainer with learning rate, decay, early stopping and more.
 - `manifold::neat::Neat` Distributed async NEAT implementation (Neuro Evolution of Augmenting Topologies) using ZMQ workers.

## Layer types:
 - `manifold::layers::Dense` Dense (fully connected) layer.

## Network types:
 - `manifold::nn::DNN` Adjustable size dense network

## Substrate types:
 - `manifold::Substrate` Basic ringbuffer substrate using a Uniform distribution. No curvature.

### TODO:
 - make hyperparameters trainable via neat as well as network breadth and depth
 - add self healing to neat async
 - Add multi-machine distributed NEAT.
 - Add curvature property to substrate, making it harder to reach edges.
 - Add CNN
