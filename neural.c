#include <stdio.h>
#include <math.h>
#include <malloc.h>

/* 
   Note to self.
   1. a func to connect two layers.

 */

typedef struct node {
  double output;
  double error;
  struct weight **weights_forward;
  struct weight **weights_backward; 
}node;

typedef struct weight {
  double weight_value;
  struct node *from_node;
  struct node *to_node;
}weight;

typedef struct layer {
  // maybe a corresponding weight, output and error matrix here, to every layer.
  // maybe link data inside matrices to data present inside nodes.
  // maybe a macro which allows operation in matrix mode, and a function to sync data between node values and matrices.
  struct node **nodes;
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

int main() {
  
    return 0;
}
