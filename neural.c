#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#define MATRIX_MODE	0
#define LIST_MODE	1

/* 
   Note to self.
   1. maybe a learning rule function which does something, or maybe not.
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

double randomiser() {
  return rand() % 1000000 / 999999.0 + 0.0000001;
}

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

double activation_function(double x) {
  // some other alternatives can be hyperbolic tangent, softmax function, or rectifier function.
  // maybe a switch case here, which accepts type of activation function to use.
  return sigmoid(x);
}

void display_weights_forward(layer *temp_layer, neuron *temp_neuron) {
  if(temp_neuron -> weights_forward == NULL)
    return;
  for(int i = 0; i < temp_layer -> next_layer -> neuron_count; i++)
    printf("%lf ", *((temp_neuron -> weights_forward)[i] -> weight));
  printf("\n");
}

void display_weights_backward(layer *temp_layer, neuron *temp_neuron) {
  if(temp_neuron -> weights_backward == NULL)
    return;
  for(int i = 0; i < temp_layer -> previous_layer -> neuron_count; i++)
    printf("%lf ", *((temp_neuron -> weights_backward)[i] -> weight));
  printf("\n");
}

void display_output_matrix(layer *l) {
  printf("Output Matrix\n");
  for(int i = 0; i < (l -> neuron_count); i++)
    printf("%lf ", (l -> output_matrix)[i]);
  printf("\n");
}

void display_error_matrix(layer *l) {
  printf("\nError Matrix\n");
  for(int i = 0; i < (l -> neuron_count); i++)
    printf("%lf ", (l -> error_matrix)[i]);
  printf("\n");
}

void display_weight_matrix(layer *from_layer) {
  if(from_layer -> next_layer == NULL)
    return;
  printf("\nWeight Matrix\n");
  for(int i = 0; i < (from_layer -> neuron_count); i++) {
    for(int j = 0; j < (from_layer -> next_layer -> neuron_count); j++)
      printf("%lf ", (from_layer -> weight_matrix)[i][j]);
    printf("\n");
  }
  printf("\n");
}

void display_layer(layer *l) {
  display_output_matrix(l);
  display_error_matrix(l);
  display_weight_matrix(l);
}

void display_net(net *network) {
  layer *temp_layer = network -> input_layer;
  int layer_count = 1;
  while(temp_layer != NULL) {
    printf("\nLayer %d\n", layer_count++);
    printf("-------\n");
    display_layer(temp_layer);
    temp_layer = temp_layer -> next_layer;
  }
}

neuron * create_neuron() {
  neuron *temp_neuron = (neuron *) malloc(sizeof(neuron));
  temp_neuron -> weights_forward = NULL;
  temp_neuron -> weights_backward = NULL;
  return temp_neuron;
}

layer * create_layer(int neurons_in_layer) {
  layer *temp_layer = (layer *) malloc(sizeof(layer));
  temp_layer -> neuron_count = neurons_in_layer;
  temp_layer -> output_matrix = (double *) malloc(neurons_in_layer * sizeof(double));
  temp_layer -> error_matrix = (double *) malloc(neurons_in_layer * sizeof(double));
  temp_layer -> weight_matrix = NULL;
  temp_layer -> neurons = NULL;
  temp_layer -> next_layer = NULL;
  temp_layer -> previous_layer = NULL;
  for(int i = 0; i < neurons_in_layer; i++) {
    neuron *temp_neuron = create_neuron();
    temp_neuron -> next_neuron = temp_layer -> neurons;
    temp_layer -> neurons = temp_neuron;
  }
  // linking neurons to their respective places in output and error matrix
  neuron *temp_neuron = temp_layer -> neurons;
  int i = 0;
  while(temp_neuron != NULL) {
    temp_neuron -> output = &(temp_layer -> output_matrix)[i];
    (temp_layer -> output_matrix)[i] = randomiser();
    temp_neuron -> error = &(temp_layer -> error_matrix)[i];
    (temp_layer -> error_matrix)[i] = randomiser();
    temp_neuron = temp_neuron -> next_neuron;
    i++;
  }
  return temp_layer;
}

void make_connections(layer *from_layer, layer *to_layer) {
  neuron *temp1 = NULL, *temp2 = NULL;
  from_layer -> next_layer = to_layer;
  to_layer -> previous_layer = from_layer;
  // generate weight matrix
  from_layer -> weight_matrix = (double **) malloc((from_layer -> neuron_count)  * sizeof(double *));
  for(int i = 0; i < (from_layer -> neuron_count); i++)
    (from_layer -> weight_matrix)[i] = (double *) malloc((to_layer -> neuron_count) * sizeof(double));
  // randomise weights in closed range (0, 1)
  // this randomisation definitely needs some imporvement later
  for(int i = 0; i < (from_layer -> neuron_count); i++)
    for(int j = 0; j < (to_layer -> neuron_count); j++)
      (from_layer -> weight_matrix)[i][j] = randomiser();
  // connecting from_layer neurons to to_layer neurons
  temp1 = from_layer -> neurons;
  int temp_from_layer_row = 0, temp_from_layer_column = 0;
  while(temp1 != NULL) {
    temp2 = to_layer -> neurons;
    temp1 -> weights_forward = (synapse **) malloc((to_layer -> neuron_count) * sizeof(synapse *));
    int synapse_count = 0;
    temp_from_layer_column = 0;
    while(temp2 != NULL) {
      synapse *temp_synapse = (synapse *) malloc(sizeof(synapse));
      temp_synapse -> from_neuron = temp1;
      temp_synapse -> to_neuron = temp2;
      // link the weight from weight_matrix to synapse
      temp_synapse -> weight = &(from_layer -> weight_matrix)[temp_from_layer_row][temp_from_layer_column++];
      (temp1 -> weights_forward)[synapse_count++] = temp_synapse;
      temp2 = temp2 -> next_neuron;
    }
    temp_from_layer_row++;
    temp1 = temp1 -> next_neuron;
  }
  // connecting to_layer neurons back to from_layer neurons
  temp2 = to_layer -> neurons;
  temp2 -> weights_forward = NULL;
  while(temp2 != NULL) {
    temp1 = from_layer -> neurons;
    temp2 -> weights_backward = (synapse **) malloc((from_layer -> neuron_count) * sizeof(synapse *));
    for(int i = 0, synapse_count = 0; i < (from_layer -> neuron_count); i++) {
      for(int j = 0; j < (to_layer -> neuron_count); j++)
	if((temp1 -> weights_forward)[j] -> to_neuron == temp2)
	  (temp2 -> weights_backward)[synapse_count++] = (temp1 -> weights_forward)[j];
      temp1 = temp1 -> next_neuron;
    }
    temp2 = temp2 -> next_neuron;    
  }
}

void add_layer_to_net(net *network, int neurons_in_layer) {
  layer *new_layer = create_layer(neurons_in_layer);
  if(network -> input_layer == NULL)
    network -> input_layer = new_layer;
  else
    make_connections(network -> output_layer, new_layer);
  network -> output_layer = new_layer;
}

net * create_network() {
  net *network = (net *) malloc(sizeof(net));
  network -> input_layer = NULL;
  network -> output_layer = NULL;
  return network;
}

void set_outputs_for_layer(layer *l, double *inputs) {
  int temp = 0;
  while(temp != (l -> neuron_count)) {
    (l -> output_matrix)[temp] = inputs[temp];
    temp++;
  }
}

void set_errors_for_layer(layer *l, double *expected) {
  int temp = 0;
  while(temp != (l -> neuron_count)) {
    (l -> error_matrix)[temp] = (expected[temp] - (l -> output_matrix)[temp]);
    temp++;
  }
}

void propagate_using_lists(net *network, double *inputs) {
  set_outputs_for_layer(network -> input_layer, inputs);
  layer *temp_layer = network -> input_layer -> next_layer;
  while(temp_layer != NULL) {
    neuron *temp_neuron = temp_layer -> neurons;
    while(temp_neuron != NULL) {
      synapse ** wb = temp_neuron -> weights_backward;
      int synapse_count = 0;
      double summation = 0;
      while(synapse_count != (temp_layer -> previous_layer -> neuron_count)) {
	summation += ((*(wb[synapse_count] -> from_neuron -> output)) * (*(wb[synapse_count] -> weight)));
	synapse_count++;
      }
      *(temp_neuron -> output) = activation_function(summation);
      temp_neuron = temp_neuron -> next_neuron;
    }
    temp_layer = temp_layer -> next_layer;
  }
}

void propagate_using_matrices(net *network, double *inputs) {
  set_outputs_for_layer(network -> input_layer, inputs);
  layer *temp_layer = network -> input_layer -> next_layer;
  while(temp_layer != NULL) {
    for(int i = 0; i < (temp_layer -> neuron_count); i++) {
      double summation = 0;
      for(int j = 0; j < (temp_layer -> previous_layer -> neuron_count); j++)
	summation += ((temp_layer -> previous_layer -> output_matrix)[j] * (temp_layer -> previous_layer -> weight_matrix)[j][i]);
      (temp_layer -> output_matrix)[i] = activation_function(summation);
    }
    temp_layer = temp_layer -> next_layer;
  }
}

void backpropagate_using_lists(net *network, double *expected) {
  set_errors_for_layer(network -> output_layer, expected);
  layer *temp_layer = (network -> output_layer) -> previous_layer;
  while(temp_layer != NULL) {
    double *total_weight_into_neuron = (double *) malloc((temp_layer -> next_layer -> neuron_count) * sizeof(double));
    neuron *temp_neuron = temp_layer -> next_layer -> neurons;
    for(int i = 0; i < (temp_layer -> next_layer -> neuron_count); i++) {
      double summation = 0;
      synapse **wb = temp_neuron -> weights_backward;
      for(int j = 0; j < (temp_layer -> neuron_count); j++)
	summation += *(wb[j] -> weight);
      total_weight_into_neuron[i] = summation;
      temp_neuron = temp_neuron -> next_neuron;
    }
    temp_neuron = temp_layer -> neurons;      
    for(int i = 0; (temp_neuron != NULL) && i < (temp_layer -> neuron_count); i++) {
      double summation = 0;
      synapse **wf = temp_neuron -> weights_forward;
      for(int j = 0; j < (temp_layer -> next_layer -> neuron_count); j++)
	summation += ((*(wf[j] -> weight)/ total_weight_into_neuron[j]) * (*(wf[j] -> to_neuron -> error)));
      *(temp_neuron -> error) = summation;
      temp_neuron = temp_neuron -> next_neuron;
    }
    // deallocate the array to prevent memory leaks
    free(total_weight_into_neuron);
    temp_layer = temp_layer -> previous_layer;
  }
}

void backpropagate_using_matrices(net *network, double *expected) {
  
}

void propogate(net *network, double *inputs) {
  if(LIST_MODE) {
    propagate_using_lists(network,inputs);
  }
  else if(MATRIX_MODE) {
    propagate_using_matrices(network, inputs);
  }  
}

void backpropagate(net *network, double *expected) {
  if(LIST_MODE) {
    backpropagate_using_lists(network, expected);
  }
  else if(MATRIX_MODE) {
    backpropagate_using_matrices(network, expected);
  }
}

double * get_inputs(net *network) {
  // use this functon to get inputs required for training or querying the neural net.
  double *inputs = (double *) malloc((network -> input_layer -> neuron_count) * sizeof(double));
  for(int i = 0; i < (network -> input_layer -> neuron_count); i++)
    inputs[i] = randomiser();
  return inputs;
}

void construct_network_architecture(net *network, int input_layers, int hidden_layers, int output_layers) {
  add_layer_to_net(network, input_layers);
  add_layer_to_net(network, hidden_layers);  
  add_layer_to_net(network, output_layers);  
}

int main() {
  net *network = create_network();
  
  construct_network_architecture(network, 2, 2, 2);
    
  double **arr = (network -> input_layer -> weight_matrix);
  arr[0][0] = 3.0;
  arr[0][1] = 1.0;
  arr[1][0] = 2.0;
  arr[1][1] = 7.0;
  arr = (network -> input_layer -> next_layer -> weight_matrix);
  arr[0][0] = 2.0;
  arr[0][1] = 1.0;
  arr[1][0] = 3.0;
  arr[1][1] = 4.0;
  

  
  //printf("\nsigmoid = %lf\n", activation_function(1.254));
  //display_net(network);
  printf("###############################\n");

  double brr[2] = {0.9, 0.1};
  propogate(network, brr);

  double *drr = (network -> output_layer -> error_matrix);
  drr[0] = 0;
  drr[1] = 0;

  double crr[2] = {2.487772, 1.486291};

  backpropagate(network, crr);
  
  display_net(network);
  return 0;
}
