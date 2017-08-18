#include <stdio.h>
#include <math.h>
#include <malloc.h>

#define MATRIX_MODE	0
#define LIST_MODE	1

/* 
   Note to self.
   1. func. to multiply, add, subtract two matrices, to multiply matrix by a scalar, to quick mupltiply a matrix with transpose of other matrix.
   2. maybe a destination matrix will also be passed to these functions to store the result.
   3. propagation and backpropagation functions
   4. maybe a learning rule function which does something, or maybe not.
*/

typedef struct neuron {
  double *output;
  double *error;
  struct synapse **weights_forward;
  struct synapse **weights_backward;
  struct neuron *next_neuron;
}neuron;

typedef struct synapse {
  double *weight;
  struct neuron *from_neuron;
  struct neuron *to_neuron;
}synapse;

typedef struct layer {
  int neuron_count;
  double *output_matrix;
  double *error_matrix;
  double **weight_matrix;
  struct neuron *neurons;
  struct layer *next_layer;
  struct layer *previous_layer;
}layer;

typedef struct net {
  struct layer *input_layer;
  struct layer *output_layer;
}net;

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

double activation_function(double x) {
  // some other alternatives can be hyperbolic tangent, softmax function, or rectifier function.
  // maybe a switch case here, which accepts type of activation function to use.
  return sigmoid(x);
}

void display_neural_network() {
  // function for debugging neural network
}

neuron * push_neuron_onto_layer(neuron *neurons) {
  neuron *temp_neuron = (neuron *) malloc(sizeof(neuron));
  temp_neuron -> weights_forward = NULL;
  temp_neuron -> weights_backward = NULL;
  temp_neuron -> next_neuron = neurons;
  neurons = temp_neuron;
  return temp_neuron;
}

layer * create_layer(int neurons_in_layer) {
  layer *temp_layer = (layer *) malloc(sizeof(layer));
  temp_layer -> neuron_count = neurons_in_layer;
  temp_layer -> output_matrix = (double *) malloc(neurons_in_layer * sizeof(double));
  temp_layer -> error_matrix = (double *) malloc(neurons_in_layer * sizeof(double));
  temp_layer -> weight_matrix = NULL;
  temp_layer -> next_layer = NULL;
  temp_layer -> previous_layer = NULL;
  for(int i = 0; i < neurons_in_layer; i++) {
    neuron *temp_neuron = push_neuron_onto_layer(temp_layer -> neurons);
    temp_neuron -> output = &(temp_layer -> output_matrix)[i];
    (temp_layer -> output_matrix)[i] = 0;
    temp_neuron -> error = &(temp_layer -> error_matrix)[i];
    (temp_layer -> error_matrix)[i] = 0;
  }
  return temp_layer;
}

void make_connections(layer *from_layer, layer *to_layer) {
  neuron *temp1 = from_layer -> neurons, *temp2 = to_layer -> neurons;
  int temp_from_layer_row = 0, temp_from_layer_column = 0;
  from_layer -> weight_matrix = (double **) malloc((from_layer -> neuron_count) * (to_layer -> neuron_count) * sizeof(double));
  while(temp1 != NULL) {
    temp2 = to_layer -> neurons;
    temp1 -> weights_forward = (synapse **) malloc((to_layer -> neuron_count) * sizeof(synapse));
    int synapse_count = 0;
    while(temp2 != NULL) {
      synapse *temp_synapse = (synapse *) malloc(sizeof(synapse));
      if(temp_from_layer_row >= from_layer -> neuron_count)
	temp_from_layer_row++;
      if(temp_from_layer_column >= to_layer -> neuron_count)
	temp_from_layer_column++;
      temp_synapse -> weight = &(from_layer -> weight_matrix)[temp_from_layer_row][temp_from_layer_column];
      temp_synapse -> from_neuron = temp1;
      temp_synapse -> to_neuron = temp2;
      temp2 = temp2 -> next_neuron;
      (temp1 -> weights_forward)[synapse_count++] = temp_synapse;
    }
    temp1 = temp1 -> next_neuron;
  }
  temp1 = to_layer -> neurons;
  temp2 -> weights_forward = NULL;
  while(temp2 != NULL) {
    temp2 -> weights_backward = (synapse **) malloc((from_layer -> neuron_count) * sizeof(synapse));
    int synapse_count = 0;
    while(temp1 != NULL) {
      for(int i = 0; i < (from_layer -> neuron_count); i++)
	if((((temp1 -> weights_forward)[i]) -> to_neuron) == temp2)
	  (temp2 -> weights_backward)[synapse_count++] = (temp1 -> weights_forward)[i];
      temp1 = temp1 -> next_neuron;
    }
    temp2 = temp2 -> next_neuron;    
  }
}

void append_new_layer_to_neural_net(net *network, int neurons_in_layer) {
  layer *new_layer = create_layer(neurons_in_layer);
  if(network -> input_layer == NULL)
    network -> input_layer = new_layer;
  else
    make_connections(network -> output_layer, new_layer);
  network -> output_layer = new_layer;
}

void set_outputs_for_layer(layer *l, double *inputs) {
  int temp = 0;
  while(temp != (l -> neuron_count))
    (l -> output_matrix)[temp] = inputs[temp];
}

void set_errors_for_layer(layer *l, double *expected) {
  int temp = 0;
  while(temp != (l -> neuron_count))
    (l -> error_matrix)[temp] = (expected[temp] - (l -> output_matrix)[temp]);
}

void propagate_using_lists(net *network, double *inputs) {
  set_outputs_for_layer(network -> input_layer, inputs);
  layer *l = (network -> input_layer) -> next_layer;
  while(l != NULL) {
    neuron *n = l -> neurons;
    while(n != NULL) {
      synapse ** wb = n -> weights_backward;
      int synapse_count = 0;
      double summation = 0;
      while(synapse_count != (l -> previous_layer -> neuron_count)) {
	summation += ((*(wb[synapse_count] -> from_neuron -> output)) * (*(wb[synapse_count] -> weight)));
	synapse_count++;
      }
      *(n -> output) = activation_function(summation);
      n = n -> next_neuron;
    }
    l = l -> next_layer;
  }
}

void backpropagate_using_lists(net *network, double *expected) {
  set_errors_for_layer(network -> output_layer, expected);
  layer *l = (network -> output_layer) -> previous_layer;
  while(l -> previous_layer == NULL) {
    neuron *n = l -> neurons;
    while(n != NULL) {
      synapse ** wf = n -> weights_forward;
      int synapse_count = 0;
      double total_error = 0;
      while(synapse_count != (l -> next_layer -> neuron_count)) {
	total_error += ((*(wf[synapse_count] -> to_neuron -> error)) * (*(wf[synapse_count] -> weight)));
	synapse_count++;
      }
      *(n -> error) = total_error;
      n = n -> next_neuron;
    }
    l = l -> next_layer;
  }  
}

void propagate_using_matrices(net *network, double *inputs) {
  
}

void backpropagate_using_matrices(net *network, double *expected) {
  
}

void propogate(net *network, double *inputs) {
  if(LIST_MODE) {
    propagate_using_lists(net *network, double *inputs);
  }
  else if(MATRIX_MODE) {
    propagate_using_matrices(net *network, double *inputs);
  }  
}

void backpropogate() {
  if(LIST_MODE) {
    backpropagate_using_lists(net *network, double *expected);
  }
  else if(MATRIX_MODE) {
    backpropagate_using_matrices(net *network, double *expected);
  }
}

int main() {
  puts("ultra!");
  return 0;
}
