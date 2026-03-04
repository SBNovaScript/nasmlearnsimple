# NASM Learn Simple — Design Document

**Date:** 2026-03-04
**Status:** Approved

## Overview

A fully NASM-based minimal machine learning library implementing dense layers with
backpropagation, proving it can encode weights and biases to solve XOR — a problem
that requires hidden layers and nonlinear activation.

## Decisions

- **100% pure NASM** — no C harness, syscalls for I/O, all math in assembly
- **SSE2 scalar doubles** — modern xmm registers, 64-bit double precision
- **Modular library** — separate .asm files per concern, linked via Makefile
- **Target:** Linux x86-64, ELF64, System V AMD64 ABI

## Project Structure

```
nasmlearnsimple/
├── Makefile
├── src/
│   ├── main.asm              # Entry point, training loop, result printing
│   ├── math/
│   │   ├── exp.asm           # e^x via range reduction + minimax polynomial
│   │   ├── sigmoid.asm       # sigmoid + sigmoid_derivative
│   │   └── dot.asm           # dot product of double arrays
│   ├── nn/
│   │   ├── forward.asm       # Dense layer forward pass
│   │   ├── backward.asm      # Backpropagation gradients
│   │   └── update.asm        # SGD weight update
│   ├── io/
│   │   ├── print.asm         # Print double, strings via sys_write
│   │   └── string.asm        # Double-to-ASCII conversion
│   └── data/
│       └── xor.asm           # XOR training set
├── include/
│   └── constants.inc         # Shared constants
└── docs/
    └── plans/
```

## Network Architecture

```
Input (2) → Hidden (2, sigmoid) → Output (1, sigmoid)
```

- **w_hidden:** 4 doubles (2x2, row-major) + 2 biases
- **w_output:** 2 doubles (1x2) + 1 bias
- **Loss:** Mean Squared Error
- **Optimizer:** Vanilla SGD, learning rate 0.5
- **Epochs:** ~10,000
- **Success:** outputs < 0.05 for (0,0)/(1,1), > 0.95 for (0,1)/(1,0)

## Function Interfaces

All follow System V AMD64 ABI. Doubles passed/returned in xmm0.

### Math

| Function | Inputs | Output |
|---|---|---|
| `exp_approx` | xmm0: x | xmm0: e^x |
| `sigmoid` | xmm0: x | xmm0: sigmoid(x) |
| `sigmoid_deriv` | xmm0: sigmoid_output | xmm0: x*(1-x) |
| `dot_product` | rdi: *A, rsi: *B, rdx: count | xmm0: result |

### Neural Network

| Function | Purpose |
|---|---|
| `forward_layer` | output[j] = sigmoid(dot(input, W[j]) + b[j]) |
| `backward_output` | Output layer gradients |
| `backward_hidden` | Hidden layer gradients via chain rule |
| `update_weights` | w[i] -= lr * grad[i] |

### I/O

| Function | Purpose |
|---|---|
| `print_double` | Double → decimal string → stdout |
| `print_newline` | Write newline |
| `print_string` | Write string of given length |

## Key Algorithms

### exp_approx

Range reduction: x = n*ln2 + r, then e^x = 2^n * P(r) where P is a degree-6
minimax polynomial accurate in [-ln2/2, ln2/2].

### Weight Initialization

LCG PRNG seeded from rdtsc, producing doubles in [-1, 1].

### print_double

Extract sign, integer part (cvttsd2si), fractional part (multiply by 10000),
print with 4 decimal places.

### Training Loop

Per epoch: iterate all 4 XOR samples, forward pass both layers, backward pass
both layers, update all weights and biases. Print loss every 1000 epochs.
