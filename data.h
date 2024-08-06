#pragma once

#include "engine.h"
#include <stddef.h>


static inline scalar Scalar(double x)
{
    return init_scalar(x);
}

static inline tensor Tensor(double *x, size_t dim)
{
    tensor t = (tensor)malloc(dim * sizeof(scalar));
    for (size_t i = 0; i < dim; i++)
    {
        t[i] = init_scalar(x[i]);
    }
    return t;
}
