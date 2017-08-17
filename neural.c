#include <stdio.h>
#include <math.h>
#include <malloc.h>

/* 
   Note to self.
   1. a func to connect two layers.

 */

typedef struct neuron {
  double *output; // points to entries inside output matrix
  double *error;// points to entries inside error matrix
  struct weight **weights_forward; // list of weights that the neuron connects to, the weight nodes will further connects to other neuron.
  struct weight **weights_backward; 
}neuron;

typedef struct weight {
  double *weight_value; // points to entries inside weight matrix
  struct neuron *from_neuron;
  struct neuron *to_neuron;
}weight;

typedef struct layer {
  // maybe a corresponding weight, output and error matrix here, to every layer.
  // link data inside neurons to point to data inside matrices inside of a layer.
  // a function to intialise 
  // 
  double **output_matrix;
  double **error_matrix;
  double **weight_matrix;	/* these are the weights after a layer, will be NULL for last layer */
  struct neuron **neurons;
  struct layer *next_layer;
  struct layer *previous_layer;
}layer;

typedef struct net {
  struct layer **layers;
  struct layer *input_layer;
  struct layer *output_layer;
}net;

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

void connect_two_layers(struct layer *layer1, struct layer layer2) {}

void append_layer_to_neural_net(net **network, int neurons_in_layer) {}

int main() {
  
    return 0;
}
