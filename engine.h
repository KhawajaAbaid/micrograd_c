#pragma once


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>


enum Op
{
    OP_ADD,
    OP_SUBTRACT,
    OP_MULITPLY,
    OP_POWER,
    OP_RELU,
    OP_ABS,
    OP_TANH
};

typedef struct Value
{
    double data;
    double grad;
    char label[5];
    enum Op _op;
    size_t _n_children;
    struct Value **_children;
    void (*_backward)(struct Value *);
    double aux;     // any auxilary value to use in backward pass
} Value;

typedef Value * scalar;    // scalar tensor
typedef Value ** tensor;     // 1d-tensor

static inline scalar init_scalar(double data)
{
    scalar s = (scalar)malloc(sizeof(Value));
    s->data = data;
    s->grad = 0.0;
    s->_backward = NULL;
    s->_children = NULL;
    s->_n_children = 0;
    return s;
}

static inline scalar init_scalar_with_children(double data,
                                               tensor children,
                                               size_t n_children)
{
    scalar s = init_scalar(data);
    s->_children = children;
    s->_n_children = n_children;
    return s;
}

static inline void free_scalar(scalar s)
{
    if (s->_n_children > 0)
    {
        for (int i = 0; i < s->_n_children; i++)
        {
            free_scalar(s->_children[i]);
        }
    }
    free(s->_children);
    free(s);
    return;
}

typedef struct
{
    scalar p;
} Unary;

typedef struct
{
    scalar p;
    scalar q;
} Binary;


static inline void backward_add(scalar self)
{
    self->_children[0]->grad += self->grad;
    self->_children[1]->grad += self->grad;
}

static inline scalar add(scalar a, scalar b)
{
    double res = a->data + b->data;
    tensor children = (tensor)malloc(2 * sizeof(scalar));
    children[0] = a;
    children[1] = b;
    scalar c = init_scalar_with_children(res, children, 2);
    c->_op = OP_ADD;
    c->_backward = &backward_add;
    return c;
}

static inline void backward_subtract(scalar self)
{
    self->_children[0]->grad += self->grad;
    self->_children[1]->grad += self->grad;
}

static inline scalar subtract(scalar a, scalar b)
{
    double res = a->data - b->data;
    tensor children = (tensor)malloc(2 * sizeof(scalar));
    children[0] = a;
    children[1] = b;
    scalar c = init_scalar_with_children(res, children, 2);
    c->_op = OP_SUBTRACT;
    c->_backward = &backward_subtract;
    return c;
}

static inline void backward_multiply(scalar self)
{
    self->_children[0]->grad += self->_children[1]->data * self->grad;
    self->_children[1]->grad += self->_children[0]->data * self->grad;
}

static inline scalar multiply(scalar a, scalar b)
{
    double res = a->data * b->data;
    tensor children = (tensor)malloc(2 * sizeof(scalar));
    children[0] = a;
    children[1] = b;
    scalar c = init_scalar_with_children(res, children, 2);
    c->_op = OP_MULITPLY;
    c->_backward = &backward_multiply;
    return c;
}

static inline void backward_power_up(scalar self)
{
    self->_children[0]->grad += (self->aux * pow(self->_children[0]->data, self->aux - 1)) * self->grad;
}

static inline scalar power_up(scalar a, double power)
{
    double res = pow(a->data, power);
    tensor children = (tensor)malloc(1 * sizeof(scalar));
    children[0] = a;
    scalar c = init_scalar_with_children(res, children, 1);
    c->_op = OP_POWER;
    c->aux = power;
    c->_backward = &backward_power_up;
    return c;
}

static inline void backward_relu(scalar self)
{
    self->_children[0]->grad += ((self->_children[0]->data > 0.0) ? 1.0 : 0.0) * self->grad;
}

static inline scalar relu(scalar a)
{
    double res = (a->data > 0.0) ? a->data : 0.0;
    tensor children = (tensor)malloc(1 * sizeof(scalar));
    children[0] = a;
    scalar c = init_scalar_with_children(res, children, 1);
    c->_op = OP_RELU;
    c->_backward = &backward_relu;
    return c;
}

static inline void backward_absolute(scalar self)
{
    self->_children[0]->grad += (self->_children[0]->data >= 0.0) ? self->grad : -self->grad;
}

static inline scalar absolute(scalar a)
{
    double res = (a->data >= 0.0) ? a->data : -a->data;
    tensor children = (tensor)malloc(1 * sizeof(scalar));
    children[0] = a;
    scalar c = init_scalar_with_children(res, children, 1);
    c->_op = OP_ABS;
    c->_backward = &backward_absolute;
    return c;
}

static inline void backward_tan_hyperbolic(scalar self)
{
    self->_children[0]->grad += (1.0 - pow(tanh(self->_children[0]->data), 2.0)) * self->grad;
}

static inline scalar tan_hyperbolic(scalar a)
{
    double res = tanh(a->data);
    tensor children = (tensor)malloc(1 * sizeof(scalar));
    children[0] = a;
    scalar c = init_scalar_with_children(res, children, 1);
    c->_op = OP_TANH;
    c->_backward = &backward_absolute;
    return c;
}

typedef struct Node
{
    scalar v;
    struct Node *next;
    struct Node *prev;
} Node;

Node *create_value_node(scalar v)
{
    Node *n = (Node *)malloc(sizeof(Node));
    n->v = v;
    n->next = NULL;
    n->prev = NULL;
    return n;
}

typedef struct
{
    Node *head;
    Node *tail;
    char name[10];
} LinkedList;


void ll_append(LinkedList *l, Node *n)
{
    if (l->head == NULL)
    {
        n->next = NULL;
        n->prev = NULL;
        l->head = n;
        l->tail = n;
        return;
    }
    n->next = NULL;
    n->prev = l->tail;
    l->tail->next = n;
    l->tail = n;
}

// Checks whether the node exists in the list 
static inline bool ll_exists(LinkedList *l, Node *n)
{
    if (l == NULL) return false;
    Node *curr = l->head;
    while (curr)
    {
        if (curr == n)
        {
            return true;
        }
        curr = curr->next;
    }
    return false;
}

static inline void free_value_linked_list(LinkedList *l)
{
    Node *curr = l->head;
    while (curr)
    {
        // free Value
        break;
    }
}

void build_topo(scalar v, LinkedList *topo, LinkedList *visited)
{
    Node *v_node = create_value_node(v);
    if (!ll_exists(visited, v_node))
    {
        ll_append(visited, v_node);
        for (int i = 0; i < v->_n_children; i++)
        {
            build_topo(v->_children[i], topo, visited);
        }
        ll_append(topo, v_node);
    }
}

void backward(scalar v)
{
    LinkedList *visited = (LinkedList *)malloc(sizeof(LinkedList));
    visited->head = NULL;
    visited->tail = NULL;
    strcpy(visited->name, "visited");
    LinkedList *topo = (LinkedList *)malloc(sizeof(LinkedList));
    topo->head = NULL;
    topo->tail = NULL;
    strcpy(topo->name, "topo");
    build_topo(v, topo, visited);
    v->grad = 1.0;
    Node *curr = topo->tail;
    while (curr)
    {
        if (curr->v->_backward != NULL)
        {
            curr->v->_backward(curr->v);
        }
        curr = curr->prev;
    }
}
