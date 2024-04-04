# Manifold

Weight separated machine learning framework.

## Network types:
 - `manifold::nn::fc::Manifold` Fully connected feedforward network.

## Trainer types:
 - `manifold::nn::Trainer` Backpropagation trainer with learning rate, decay, early stopping and more.
 - `manifold::async_neat::Neat` Distributed async NEAT implementation (Neuro Evolution of Augmenting Topologies) - mad fast.
 - `manifold::sync_neat::Neat` Distributed sync NEAT implementation (Neuro Evolution of Augmenting Topologies) - slower.

## Substrate types:
 - `manifold::Substrate` Basic ringbuffer substrate using a Uniform distribution. No curvature.

### TODO:
 - [done] Add mutli-core distributed NEAT.
 - [done] Make manifolds serializable into binary.
 - [done] Make substrates serializable into binary.
 - [bad idea] Make neat that tests all architectures in a range.
 - [done] Make flow-state neat, where worker state doesnt have to be synced. 
 - [done] Pass generic early stopping params to neat.
 - Add multi-machine distributed NEAT.
 - Add curvature property to substrate, making it harder to reach edges.
