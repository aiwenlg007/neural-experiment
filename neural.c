#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#define LEARNING_RATE	0.3
#define MATRIX_MODE	1
#define LIST_MODE	0
#define SHOW_STATS	1

/* 
   Note to self.
   1. an inverse activation function
   2. logic to reverse flow of things to know neural net's viewpoint

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
  return rand() % 1000000 / 999998.0 + 0.000001;
}

double learning_rate() {
  return LEARNING_RATE;
}

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

double activation_function(double x) {
  // some other alternatives can be hyperbolic tangent, softmax function, or rectifier function.
  // maybe a switch case here, which accepts type of activation function to use.
  return sigmoid(x);
}

double gradient_of_activaton_function(double error_of_synapse_from_back_layer, double output_of_synapse_from_layer, double output_of_synapse_to_front_layer) {
  // currently only for sigmoid/logistic functon
  return -1 * learning_rate() * error_of_synapse_from_back_layer   * sigmoid(output_of_synapse_to_front_layer) * sigmoid(1 - output_of_synapse_to_front_layer) * output_of_synapse_from_layer;
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
    (temp_layer -> output_matrix)[i] = 0;
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
  // randomise weights in some range
  for(int i = 0; i < (from_layer -> neuron_count); i++)
    for(int j = 0; j < (to_layer -> neuron_count); j++)
      (from_layer -> weight_matrix)[i][j] = randomiser() * 2 - 1;
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
    (l -> error_matrix)[temp] = ((l -> output_matrix)[temp]- expected[temp]);
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
    // calculate total weight going into each neuron of next layer
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
    // backpropagate errors
    temp_neuron = temp_layer -> neurons;      
    for(int i = 0; (temp_neuron != NULL) && i < (temp_layer -> neuron_count); i++) {
      double summation = 0;
      synapse **wf = temp_neuron -> weights_forward;
      for(int j = 0; j < (temp_layer -> next_layer -> neuron_count); j++)
	summation += ((*(wf[j] -> weight)/ total_weight_into_neuron[j]) * (*(wf[j] -> to_neuron -> error)));
      *(temp_neuron -> error) = summation;
      temp_neuron = temp_neuron -> next_neuron;
    }
    // deallocate the allocated array to prevent memory leaks
    free(total_weight_into_neuron);
    // update weights
    temp_neuron = temp_layer -> neurons;
    while(temp_neuron != NULL) {
      synapse **wf = temp_neuron -> weights_forward;
      for(int i = 0; i < (temp_layer -> next_layer -> neuron_count); i++)
	*(wf[i] -> weight) = *(wf[i] -> weight) - learning_rate() * gradient_of_activaton_function( *(wf[i] -> from_neuron -> error), *(wf[i] -> from_neuron -> output), *(wf[i] -> to_neuron -> output));
      temp_neuron = temp_neuron -> next_neuron;
    }
    temp_layer = temp_layer -> previous_layer;
  }
}

void backpropagate_using_matrices(net *network, double *expected) {
  set_errors_for_layer(network -> output_layer, expected);
  layer *temp_layer = (network -> output_layer) -> previous_layer;
  while(temp_layer != NULL) {
    // backpropagate errors
    for(int i = 0; i < (temp_layer -> neuron_count); i++) {
      (temp_layer -> error_matrix)[i] = 0;
      for(int j = 0; j < (temp_layer -> next_layer -> neuron_count); j++) {
	double summation = 0;
	for(int k = 0; k < (temp_layer -> neuron_count); k++)
	  summation += (temp_layer -> weight_matrix)[k][j];
	(temp_layer -> error_matrix)[i] += (temp_layer -> weight_matrix)[i][j] * (temp_layer -> next_layer -> error_matrix)[j] / summation;  
      }
    }
    // update weights
    for(int i = 0; i < (temp_layer -> neuron_count); i++)
      for(int j = 0; j < (temp_layer -> next_layer -> neuron_count); j++)
	//(temp_layer -> weight_matrix)[i][j] += (learning_rate() * gradient_of_activaton_function((temp_layer -> next_layer -> error_matrix)[i], (temp_layer -> next_layer -> output_matrix)[i], (temp_layer -> output_matrix)[j]));
	(temp_layer -> weight_matrix)[i][j] += (learning_rate() * gradient_of_activaton_function((temp_layer -> error_matrix)[i], (temp_layer -> output_matrix)[i], (temp_layer -> next_layer -> output_matrix)[j]));
    temp_layer = temp_layer -> previous_layer;
  }  
}

void propagate(net *network, double *inputs) {
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

void construct_network_architecture(net *network, int input_layers, int hidden_layers, int output_layers) {
  add_layer_to_net(network, input_layers);
  add_layer_to_net(network, hidden_layers);  
  add_layer_to_net(network, output_layers);  
}

double * get_random_inputs(net *network) {
  double *inputs = (double *) malloc((network -> input_layer -> neuron_count) * sizeof(double));
  for(int i = 0; i < (network -> input_layer -> neuron_count); i++)
    inputs[i] = randomiser();
  return inputs;
}

int get_label() {
  int num;
  fscanf(stdin, "%d,", &num);
  //printf("\nlabel = %d\n", num);
  return num;
}

double * get_expected() {
  double *expected = (double *) malloc(10 * sizeof(double));
  for(int i = 0; i < 10; i++)
    expected[i] = 0.01;
  expected[get_label()] = 0.99;
  return expected;
}

double * get_inputs() {
  // use this functon to get inputs required for training or querying the neural net.
  double *inputs = (double *) malloc(784 * sizeof(double));
  for(int i = 0; i < 784; i++) {
    fscanf(stdin, "%lf,", &inputs[i]);
    inputs[i] = (inputs[i]) / 255.0 * 0.99 + 0.01;
  }    
  return inputs;
}

int query(net *network) {
  int label = get_label();
  double *inputs = get_inputs();
  propagate(network, inputs);
  int max_label = 0;
  double max_score = (network -> output_layer -> output_matrix)[0];
  for(int i = 1; i < (network -> output_layer -> neuron_count); i++) {
    if((network -> output_layer -> output_matrix)[i] > max_score) {
      max_label = i;
      max_score = (network -> output_layer -> output_matrix)[i];
    }
  }
  if(SHOW_STATS) {
    printf("\nlabel = %d -> %d\n", label, max_label);
    if(SHOW_STATS) {
       if(label == max_label)
	 printf("CORRECT\n");
       else
      printf("INCORRECT\n");
    }
    for(int i = 0; i < (network -> output_layer -> neuron_count); i++)
      printf("%d -> %.0lf\n", i, (network -> output_layer -> output_matrix)[i] * 100);
  }
  if(label == max_label)
    return 1;
  else
    return 0;
  free(inputs);
}

void train(net *network) {
  double *expected = get_expected();
  double *inputs = get_inputs();
  propagate(network, inputs);
  backpropagate(network, expected);
  // experiments
  //puts("");
  //display_output_matrix(network -> output_layer);
  free(expected);
  free(inputs);
}

int main() {
  net *network = create_network();

  construct_network_architecture(network, 784, 100, 10);
  
  int t = 10;
  int q = 1200;
  double score = 0;
  
  // train the network
  freopen("mnist_train.csv", "r", stdin);
  for(int i = 0; i < t; i++) {
    train(network);
    //display_error_matrix(network -> output_layer);
    //display_output_matrix(network -> output_layer);
  }
     
  // query the network
  freopen("mnist_test.csv", "r", stdin);
  for(int i = 0; i < q; i++) {
    score += query(network);
  }
  printf("\nAccuracy = %lf", (score / q * 100));
  
  return 0;
}
