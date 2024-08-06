#include "engine.h"
#include "nn.h"
#include "data.h"


void print_tensor(tensor t, size_t dim)
{
    printf("Tensor([");
    for (size_t i = 0; i < dim; i++)
    {
        printf("%f%s", t[i]->data, ((i+1) == dim) ? "])\n" : ", ");
    }
}

void print_tensor_grads(tensor t, size_t dim)
{
    printf("Tensor([");
    for (size_t i = 0; i < dim; i++)
    {
        printf("%f%s", t[i]->grad, ((i+1) == dim) ? "])\n" : ", ");
    }
}

int main()
{
    const size_t n_layers = 3;
    const size_t n_in = 3;
    size_t n_outs[3] = {5, 5, 1};
    const double learning_rate = 0.0001;
    enum Activation act_hidden = ACT_TANH;
    enum Activation act_out = ACT_RAW_PLEASE;
    MLP *mlp = init_mlp(n_in, n_outs, n_layers, act_hidden, act_out);
    
    const size_t n_samples = 4;
    double xs[4][3] = {
        {-0.07708825,  1.09136604, -1.47771791},
        {0.46909754,  1.45333126,  0.21135764},
        {0.46909754,  1.45333126,  0.21135764},
        {1.78757578, -0.87620064,  0.48024694}
    };
    
    tensor *xs_t = (tensor *)malloc(n_samples * sizeof(tensor));
    for (size_t i = 0; i < n_samples; i++)
    {
        xs_t[i] = Tensor(xs[i], 3);
    }
    double ys[4] = {1.0, -1.0, -1.0, 1.0};
    tensor ys_t = Tensor(ys, 4);
    
    const size_t n_epochs = 20;
    tensor y_pred;
    scalar y_preds[4];
    scalar loss;
    tensor params;
    for (size_t epoch = 0; epoch < n_epochs; epoch++)
    {
        // forward pass
        loss = init_scalar(0.0);
        for (size_t i = 0; i < n_samples; i++)
        {
            y_pred = apply_mlp(mlp, xs_t[i]);
            y_preds[i] = y_pred[0];
            free(y_pred);
            loss = add(loss, absolute(subtract(ys_t[i], y_preds[i])));
        }

        // backward
        backward(loss);
        params = get_mlp_params(mlp);
        update_params(mlp, learning_rate);

        printf("Epoch: %ld | Loss: %.5f\n", epoch, loss->data);
    }
    return 0;
}

