#pragma once

#include "engine.h"
#include <float.h>
#include <math.h>
#include "random.h"


typedef struct
{
    tensor w;
    scalar b;
    size_t n_in;
} Neuron;

static inline Neuron *init_neuron(const size_t n_in, const size_t n_out)
{
    Neuron *n = (Neuron *)malloc(sizeof(Neuron));
    n->w = (tensor)malloc(n_in * sizeof(scalar));
    for (size_t i = 0; i < n_in; i++)
    {
        n->w[i] = init_scalar(glorot_random_normal(n_in, n_out));
    }
    n->b = init_scalar(0.);
    n->n_in = n_in;
    return n;
}

static inline scalar apply_neuron(Neuron *n, tensor x)
{
    scalar logit = init_scalar(0.0);
    for (size_t i = 0; i < n->n_in; i++)
    {
        logit = add(logit, multiply(n->w[i], x[i]));
        logit = add(logit, n->b);
    }
    return logit;
}

static inline tensor get_neuron_params(Neuron *n)
{
    tensor params = (tensor)malloc((n->n_in+1) * sizeof(scalar));
    for (size_t i = 0; i < n->n_in; i++)
    {
        params[i] = n->w[i];
    }
    params[n->n_in] = n->b;
    return params;
}

enum Activation
{
    ACT_RAW_PLEASE,
    ACT_RELU
};

typedef struct
{
    Neuron **ns;
    size_t n_in;
    size_t n_out;
    enum Activation act;
} Layer;

static inline Layer *init_layer(size_t n_in, size_t n_out, enum Activation act)
{
    Layer *l = (Layer *)malloc(sizeof(Layer));
    l->ns = (Neuron **)malloc(n_out * sizeof(Neuron *));
    for (size_t i = 0; i < n_out; i++)
    {
        l->ns[i] = init_neuron(n_in, n_out);
    }
    l->n_in = n_in;
    l->n_out = n_out;
    l->act = act;
    return l;
}

static inline tensor apply_layer(Layer *l, tensor x)
{
    tensor logits = (tensor)malloc(l->n_out * sizeof(scalar));
    for (size_t i = 0; i < l->n_out; i++)
    {
       logits[i] = apply_neuron(l->ns[i], x);
       switch (l->act)
       {
           case ACT_RAW_PLEASE:
               break;
           case ACT_RELU:
               logits[i] = relu(logits[i]);
               break;
           default:
               break;
       }
    }
    return logits;
}

static inline size_t count_layer_params(Layer *l)
{
    return (l->n_in * l->n_out) + l->n_out;
}

static inline tensor get_layer_params(Layer *l)
{
    size_t n_params = count_layer_params(l);
    tensor params = (tensor)malloc(n_params * sizeof(scalar));
    tensor neuron_params;
    for (size_t i = 0; i < l->n_out; i++)
    {
        neuron_params = get_neuron_params(l->ns[i]);
        for (size_t j = 0; j <= l->n_in; j++)
        {
            params[(i * (l->n_in+1)) + j] = neuron_params[j];
        }
        free(neuron_params);
    }
    return params;
}

typedef struct
{
    Layer **layers;
    size_t n_in;
    size_t *n_outs;
    size_t n_layers;
} MLP;

static inline MLP *init_mlp(size_t n_in,
                            size_t *n_outs,
                            size_t n_layers,
                            enum Activation hidden_act,
                            enum Activation out_act)
{
    MLP *mlp = (MLP *)malloc(sizeof(MLP));
    mlp->layers = (Layer **)malloc(n_layers * sizeof(Layer *));
    for (size_t i = 0; i < n_layers; i++)
    {
        mlp->layers[i] = init_layer((i == 0) ? n_in : n_outs[i - 1],
                                    n_outs[i],
                                    ((i+1) == n_layers) ? out_act : hidden_act);
    }
    mlp->n_in = n_in;
    mlp->n_outs = n_outs;
    mlp->n_layers = n_layers;
    return mlp;
}

static inline tensor apply_mlp(MLP *mlp, tensor x)
{
    tensor logits = x;
    tensor temp = logits;
    for (size_t i = 0; i < mlp->n_layers; i++)
    {
        logits = apply_layer(mlp->layers[i], logits);
        if (temp != x) free(temp);
        temp = logits;
    }
    return logits;
}

static inline size_t count_mlp_params(MLP *mlp)
{
    size_t n_params = 0;
    for (size_t i = 0; i < mlp->n_layers; i++)
    {
        n_params += count_layer_params(mlp->layers[i]);
    }
    return n_params;
}

static inline tensor get_mlp_params(MLP *mlp)
{
    // Count total params
    size_t n_params = 0;
    size_t *n_layers_params = (size_t *)malloc(mlp->n_layers * sizeof(size_t)); 
    size_t curr_count;
    for (size_t i = 0; i < mlp->n_layers; i++)
    {
        curr_count = count_layer_params(mlp->layers[i]);
        n_params += curr_count;
        n_layers_params[i] = curr_count;
    }
    
    // Put each layer params in one params array
    tensor params = (tensor)malloc(n_params * sizeof(scalar));
    tensor layer_params;
    size_t offset = 0;
    for (size_t j = 0; j < mlp->n_layers; j++)
    {
        layer_params = get_layer_params(mlp->layers[j]);
        
        for (size_t k = 0; k < n_layers_params[j]; k++)
        {
            params[k + offset] = layer_params[k];
        }
        offset += n_layers_params[j];
    }
    return params;
}

static inline void update_params(MLP *mlp,
                                 double learning_rate)
{
    tensor params = get_mlp_params(mlp);
    size_t n_params = count_mlp_params(mlp);

    for (size_t i = 0; i < n_params; i++)
    {
        params[i]->data -= learning_rate * params[i]->grad;

        // reset grad
        params[i]->grad = 0.0;
    }
}
