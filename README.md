# Manifold

Weight separated machine learning framework.

## Network types:
 - `manifold::nn::fc::Manifold` Fully connected feedforward network.

## Trainer types:
 - `manifold::nn::Trainer` Backpropagation trainer with learning rate, decay, early stopping and more.
 - `manifold::Neat` Monte-carlo - for now - distributed NEAT implementation (Neuro Evolution of Augmenting Topologies)

## Substrate types:
 - `manifold::Substrate` Basic ringbuffer substrate using a Uniform distribution. No curvature.

### TODO:
 - [done] Add mutli-core distributed NEAT.
 - [done] Make manifolds serializable into binary.
 - [done] Make substrates serializable into binary.
 - Add multi-machine distributed NEAT.
 - Add curvature property to substrate, making it harder to reach edges.
