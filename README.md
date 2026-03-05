# NASM Learn Simple

A neural network that learns XOR — written entirely in x86-64 NASM assembly. No C, no libc, no external libraries. Just raw syscalls, registers, and floating-point math.

## What It Does

Trains a 2-2-1 feedforward neural network to solve the XOR problem using backpropagation with online stochastic gradient descent. The entire pipeline — forward pass, loss computation, gradient calculation, weight updates, I/O, and timing — is implemented in ~1500 lines of hand-written assembly.

```
$ ./build/nasmlearn
=== NASM Learn Simple ===
Training XOR network (2-2-1)...
Epoch 0 | Loss: 0.3007
Epoch 1000 | Loss: 0.0014
Epoch 2000 | Loss: 0.0004
Epoch 3000 | Loss: 0.0003
Epoch 4000 | Loss: 0.0002
Epoch 5000 | Loss: 0.0001
Epoch 6000 | Loss: 0.0001
Epoch 7000 | Loss: 0.0001
Epoch 8000 | Loss: 0.0001
Epoch 9000 | Loss: 0.0001

--- Final Predictions ---
Input: (0.0000, 0.0000) -> 0.0074 (expected: 0.0000)
Input: (0.0000, 1.0000) -> 0.9923 (expected: 1.0000)
Input: (1.0000, 0.0000) -> 0.9922 (expected: 1.0000)
Input: (1.0000, 1.0000) -> 0.0095 (expected: 0.0000)

--- Timing Results ---
Trials:  10
Average: 3.3150 ms
Min:     2.8011 ms
Max:     6.2643 ms
```

## Build & Run

**Requirements:** NASM assembler, GNU `ld` linker, Linux x86-64.

```bash
make clean && make
./build/nasmlearn
```

The Makefile auto-discovers all `.asm` files under `src/` — no manual file lists to maintain.

## Architecture

### Network

| Layer | Neurons | Activation |
|-------|---------|------------|
| Input | 2 | — |
| Hidden | 2 | Sigmoid |
| Output | 1 | Sigmoid |

- **Training:** Online SGD (one sample at a time, weights updated after each)
- **Learning rate:** 2.0
- **Loss:** Mean squared error (factor of 2 omitted, absorbed into learning rate)
- **Epochs:** Up to 10,000
- **Weight init:** LCG PRNG seeded from TSC, range [-0.5, 0.5) (Xavier-like for fan_in=2)

### Convergence Strategy

About 10% of random initializations land in a local minimum where the loss plateaus above the threshold. The program handles this automatically: if loss > 0.01 after all epochs, it reinitializes weights and restarts training. This guarantees convergence on every run.

### System-Level Details

- **Target:** Linux x86-64 ELF64
- **ABI:** System V AMD64 calling convention
- **Floating point:** SSE2 scalar doubles (`movsd`, `addsd`, `mulsd`, `divsd`)
- **I/O:** Direct `sys_write` syscalls (no libc, no `printf`)
- **Timing:** `clock_gettime(CLOCK_MONOTONIC)` via syscall 228
- **Stack:** Non-executable (`GNU-stack` note in every object file)

## Project Structure

```
nasmlearnsimple/
├── include/
│   └── constants.inc          # Network dims, training params, syscall numbers
├── src/
│   ├── main.asm               # Entry point, weight init, training loop, trial loop
│   ├── math/
│   │   ├── exp.asm            # e^x via range reduction + degree-6 Horner polynomial
│   │   ├── sigmoid.asm        # sigmoid(x) = 1/(1+e^(-x)) and derivative s*(1-s)
│   │   └── dot.asm            # Dot product of two double arrays
│   ├── nn/
│   │   ├── forward.asm        # Dense layer forward pass (dot + bias + sigmoid)
│   │   ├── backward.asm       # Backprop gradients for output and hidden layers
│   │   └── update.asm         # SGD weight update: w -= lr * grad
│   ├── io/
│   │   ├── print.asm          # print_string, print_double, print_newline (syscalls)
│   │   └── string.asm         # double-to-ASCII conversion (4 decimal places)
│   ├── data/
│   │   └── xor.asm            # XOR training set (4 input/target pairs)
│   └── timing/
│       └── clock.asm          # Wall-clock timing with nanosecond precision
└── Makefile
```

## Module Details

### Math (`src/math/`)

**`exp.asm`** — Approximates e^x using the identity e^x = 2^n * e^r where n = round(x / ln2) and r = x - n*ln2. The fractional part e^r is evaluated with a degree-6 Taylor polynomial via Horner's method. Input is clamped to [-709, 709] to prevent IEEE 754 overflow/underflow. The 2^n factor is constructed by directly setting the exponent bits of a double.

**`sigmoid.asm`** — Computes sigmoid(x) = 1 / (1 + e^(-x)) by calling `exp_approx`. Also exports `sigmoid_deriv(s) = s * (1-s)` which takes the sigmoid *output* (not raw x) — a common optimization that avoids recomputing the exponential.

**`dot.asm`** — Scalar dot product loop over two `double` arrays. Leaf function with no stack frame.

### Neural Network (`src/nn/`)

**`forward.asm`** — Generic dense layer forward pass. For each output neuron: computes dot product of input with weight row, adds bias, applies sigmoid. Works for any input/output dimensions.

**`backward.asm`** — Backpropagation gradient computation. `backward_output` computes the output layer delta as (output - expected) * sigmoid'(output) and produces weight/bias gradients. `backward_hidden` propagates the output delta through the output weights, multiplies by sigmoid'(hidden), and produces hidden layer weight/bias gradients. The 7th argument (delta_hidden pointer) is passed on the stack per SysV AMD64 convention.

**`update.asm`** — Applies SGD: `w[i] -= lr * grad[i]`. Leaf function, 6 instructions.

### I/O (`src/io/`)

**`print.asm`** — Thin wrappers around `sys_write` (syscall 1) for strings, doubles, and newlines. No buffering.

**`string.asm`** — Converts a `double` to decimal ASCII with 4 fixed decimal places. Handles negative numbers, extracts integer/fractional parts separately, uses `roundsd` for correct rounding.

### Timing (`src/timing/`)

**`clock.asm`** — Performance measurement using `clock_gettime(CLOCK_MONOTONIC)`. Runs 10 trials per execution, collecting min/max/sum statistics. Reports average, minimum, and maximum convergence time in milliseconds. Epoch output is suppressed for non-final trials to avoid I/O overhead skewing results.

## Performance

Timing results from multiple runs on WSL2 (Linux 5.15, x86-64):

| Run | Average | Min | Max |
|-----|---------|-----|-----|
| 1 | 4.83 ms | 2.71 ms | 9.30 ms |
| 2 | 3.41 ms | 2.73 ms | 5.76 ms |
| 3 | 3.11 ms | 2.72 ms | 5.47 ms |
| 4 | 3.37 ms | 2.70 ms | 5.54 ms |
| 5 | 2.82 ms | 2.73 ms | 3.06 ms |
| 6 | 3.32 ms | 2.80 ms | 6.26 ms |

Each run executes 10 trials (full training to convergence). The consistent ~2.7 ms minimum represents the best-case convergence time when random initialization lands directly in a good loss basin. Higher maximums occur when a trial hits a local minimum and requires weight reinitialization before converging.

## Register Conventions

The training loop in `main.asm` uses callee-saved registers to persist state across function calls:

| Register | Purpose |
|----------|---------|
| `r12` | Epoch counter |
| `r13` | Sample index (0-3) |
| `r14` | Current input pointer |
| `r15` | Trial counter (0-9) |

All functions follow the System V AMD64 ABI: arguments in `rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`, float args in `xmm0`-`xmm7`, return values in `rax`/`xmm0`. Stack is 16-byte aligned before every `call` instruction.

## Design Decisions

**No libc:** All I/O goes through raw Linux syscalls. This keeps the binary small, eliminates dynamic linking, and demonstrates low-level system programming.

**Scalar SSE2:** Uses `movsd`/`addsd`/`mulsd`/`divsd` rather than x87 FPU or SIMD vectorization. SSE2 is baseline on all x86-64 CPUs and provides clean, predictable IEEE 754 double arithmetic.

**Online SGD:** Weights are updated after every single training sample rather than batching. For a 4-sample dataset this makes no practical difference, but it simplifies the code by avoiding gradient accumulation buffers.

**Learning rate 2.0:** Higher than typical because the MSE gradient's factor of 2 is omitted. The effective learning rate is equivalent to 1.0 with the standard MSE formulation.

**Position-independent code:** All data references use `[rel ...]` addressing (`default rel`), making the code PIC/PIE compatible even though it's linked as a static executable.
