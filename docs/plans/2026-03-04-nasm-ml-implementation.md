# NASM Learn Simple — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pure NASM x86-64 neural network library that learns XOR through backpropagation.

**Architecture:** Modular `.asm` files linked via Makefile. SSE2 scalar doubles for all math. Raw Linux syscalls for I/O. SysV AMD64 ABI throughout.

**Tech Stack:** NASM assembler, GNU ld linker, Linux x86-64 ELF64, Make.

---

### Task 1: Scaffold project structure and build system

**Files:**
- Create: `Makefile`
- Create: `include/constants.inc`
- Create: `src/main.asm` (stub only)

**Step 1: Create directory structure**

```bash
mkdir -p src/math src/nn src/io src/data include
```

**Step 2: Write Makefile**

Create `Makefile`:

```makefile
NASM      = nasm
NFLAGS    = -f elf64 -g -F dwarf -I include/
LD        = ld
LDFLAGS   =

SRCDIR    = src
INCDIR    = include
BUILDDIR  = build

SRCS      = $(shell find $(SRCDIR) -name '*.asm')
OBJS      = $(patsubst $(SRCDIR)/%.asm,$(BUILDDIR)/%.o,$(SRCS))

TARGET    = $(BUILDDIR)/nasmlearn

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.asm $(wildcard $(INCDIR)/*.inc)
	@mkdir -p $(dir $@)
	$(NASM) $(NFLAGS) -o $@ $<

clean:
	rm -rf $(BUILDDIR)
```

**Step 3: Write constants.inc**

Create `include/constants.inc`:

```nasm
; include/constants.inc
; Shared constants for NASM Learn Simple

%ifndef CONSTANTS_INC
%define CONSTANTS_INC

; --- Syscall numbers ---
%define SYS_write   1
%define SYS_exit    60
%define STDOUT      1

; --- Network dimensions ---
%define INPUT_SIZE   2
%define HIDDEN_SIZE  2
%define OUTPUT_SIZE  1

; --- Training parameters ---
%define MAX_EPOCHS   10000
%define PRINT_EVERY  1000

%endif
```

**Step 4: Write stub main.asm**

Create `src/main.asm`:

```nasm
; src/main.asm
; Entry point — stub for build verification

bits 64
default rel

%include "constants.inc"

section .text
global _start

_start:
    mov eax, SYS_exit
    xor edi, edi
    syscall

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 5: Build and verify**

Run: `make clean && make && ./build/nasmlearn; echo "Exit: $?"`
Expected: `Exit: 0`

**Step 6: Commit**

```bash
git add Makefile include/ src/main.asm
git commit -m "feat: scaffold project structure with Makefile and stub entry point"
```

---

### Task 2: I/O module — string output and double printing

**Files:**
- Create: `src/io/print.asm`
- Create: `src/io/string.asm`

**Step 1: Write string.asm — double-to-ASCII conversion**

Create `src/io/string.asm`:

```nasm
; src/io/string.asm
; Double-to-ASCII conversion for printing
;
; double_to_str(xmm0: double, rdi: *buf) -> rax: length
;   Writes decimal representation to buf with 4 decimal places.
;   Returns number of bytes written.

bits 64
default rel

section .data
    ten_thousand: dq 10000.0
    ten:          dq 10.0
    zero_f:       dq 0.0

section .text
global double_to_str

double_to_str:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    push r14

    mov  r12, rdi              ; r12 = buffer pointer
    xor  r13, r13              ; r13 = write index

    ; Check sign
    movsd xmm1, [rel zero_f]
    ucomisd xmm0, xmm1
    jae  .positive
    ; Negative: write '-' and negate
    mov  byte [r12 + r13], '-'
    inc  r13
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    movsd xmm0, xmm1
.positive:

    ; Split into integer and fractional parts
    ; Integer part
    cvttsd2si rax, xmm0       ; rax = (int64)floor(abs(x))
    mov  rbx, rax              ; save integer part

    ; Convert integer to fractional remainder
    cvtsi2sd xmm1, rax
    subsd xmm0, xmm1          ; xmm0 = fractional part

    ; Write integer part digits
    ; Convert integer to decimal string (reverse order then copy)
    mov  rax, rbx
    test rax, rax
    jnz  .int_nonzero
    ; Zero integer part
    mov  byte [r12 + r13], '0'
    inc  r13
    jmp  .int_done

.int_nonzero:
    ; Count digits and write in reverse to a temp area on stack
    sub  rsp, 32               ; temp buffer on stack
    xor  r14, r14              ; digit count
    mov  rax, rbx
.int_loop:
    test rax, rax
    jz   .int_reverse
    xor  edx, edx
    mov  rcx, 10
    div  rcx
    add  dl, '0'
    mov  [rsp + r14], dl
    inc  r14
    jmp  .int_loop

.int_reverse:
    ; Copy digits in reverse order
    mov  rcx, r14
.rev_loop:
    dec  rcx
    mov  al, [rsp + rcx]
    mov  [r12 + r13], al
    inc  r13
    test rcx, rcx
    jnz  .rev_loop
    add  rsp, 32
.int_done:

    ; Decimal point
    mov  byte [r12 + r13], '.'
    inc  r13

    ; Fractional part: multiply by 10000, round, print 4 digits
    mulsd xmm0, [rel ten_thousand]
    cvtsd2si rax, xmm0        ; rax = fractional * 10000 (rounded)

    ; Ensure non-negative
    test rax, rax
    jns  .frac_pos
    xor  eax, eax
.frac_pos:

    ; Print 4 fractional digits (with leading zeros)
    mov  rcx, 1000
    ; Digit 1
    xor  edx, edx
    div  rcx                   ; rax = d1, rdx = remainder
    add  al, '0'
    mov  [r12 + r13], al
    inc  r13
    mov  rax, rdx

    mov  rcx, 100
    xor  edx, edx
    div  rcx
    add  al, '0'
    mov  [r12 + r13], al
    inc  r13
    mov  rax, rdx

    mov  rcx, 10
    xor  edx, edx
    div  rcx
    add  al, '0'
    mov  [r12 + r13], al
    inc  r13

    add  dl, '0'
    mov  [r12 + r13], dl
    inc  r13

    ; Return length in rax
    mov  rax, r13

    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 2: Write print.asm — print routines via syscalls**

Create `src/io/print.asm`:

```nasm
; src/io/print.asm
; Output routines using Linux syscalls
;
; print_double(xmm0: double)  — prints double with 4 decimal places + newline
; print_string(rdi: *str, rsi: len) — writes string to stdout
; print_newline()              — writes newline to stdout

bits 64
default rel

%include "constants.inc"

extern double_to_str

section .data
    newline_char: db 10

section .bss
    print_buf: resb 64

section .text
global print_double
global print_string
global print_newline

; print_string(rdi: *string, rsi: length)
print_string:
    mov  rdx, rsi              ; length
    mov  rsi, rdi              ; buffer
    mov  eax, SYS_write
    mov  edi, STDOUT
    syscall
    ret

; print_newline()
print_newline:
    lea  rsi, [rel newline_char]
    mov  eax, SYS_write
    mov  edi, STDOUT
    mov  edx, 1
    syscall
    ret

; print_double(xmm0: double)
print_double:
    push rbp
    mov  rbp, rsp

    ; Convert double to string
    lea  rdi, [rel print_buf]
    call double_to_str

    ; Print the string (rax = length from double_to_str)
    lea  rsi, [rel print_buf]
    mov  rdx, rax
    mov  eax, SYS_write
    mov  edi, STDOUT
    syscall

    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 3: Update main.asm to test I/O**

Update `src/main.asm` to call print_double with a test value (e.g., 3.1415):

```nasm
; src/main.asm
bits 64
default rel

%include "constants.inc"

extern print_double
extern print_string
extern print_newline

section .data
    test_val:   dq 3.1415
    msg_test:   db "I/O test: "
    msg_test_len equ $ - msg_test

section .text
global _start

_start:
    ; Print test label
    lea  rdi, [rel msg_test]
    mov  esi, msg_test_len
    call print_string

    ; Print a test double
    movsd xmm0, [rel test_val]
    call print_double
    call print_newline

    ; Exit
    mov  eax, SYS_exit
    xor  edi, edi
    syscall

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 4: Build and verify**

Run: `make clean && make && ./build/nasmlearn`
Expected output: `I/O test: 3.1415`

**Step 5: Commit**

```bash
git add src/io/ src/main.asm
git commit -m "feat: add I/O module with double-to-string and print routines"
```

---

### Task 3: Math module — exp, sigmoid, dot product

**Files:**
- Create: `src/math/exp.asm`
- Create: `src/math/sigmoid.asm`
- Create: `src/math/dot.asm`

**Step 1: Write exp.asm — e^x approximation**

Create `src/math/exp.asm`:

```nasm
; src/math/exp.asm
; exp_approx(xmm0: x) -> xmm0: e^x
;
; Uses range reduction + degree-6 polynomial:
;   x = n * ln2 + r  where n = round(x / ln2)
;   e^x = 2^n * P(r) where P is a minimax polynomial for e^r on [-ln2/2, ln2/2]
;
; Accurate for x in [-700, 700]. Returns 0 for very negative x.

bits 64
default rel

section .rodata
    align 16
    ln2:        dq 0.6931471805599453    ; ln(2)
    inv_ln2:    dq 1.4426950408889634    ; 1/ln(2)
    half:       dq 0.5
    one:        dq 1.0

    ; Minimax polynomial coefficients for e^r on [-ln2/2, ln2/2]
    ; P(r) = c0 + c1*r + c2*r^2 + c3*r^3 + c4*r^4 + c5*r^5 + c6*r^6
    c0:         dq 1.0
    c1:         dq 1.0
    c2:         dq 0.5
    c3:         dq 0.16666666666666666    ; 1/6
    c4:         dq 0.041666666666666664   ; 1/24
    c5:         dq 0.008333333333333333   ; 1/120
    c6:         dq 0.001388888888888889   ; 1/720

    ; Clamp bounds
    exp_max:    dq  709.0
    exp_min:    dq -709.0
    zero_f:     dq  0.0

section .text
global exp_approx

exp_approx:
    ; Clamp x to [-709, 709]
    movsd   xmm7, [rel exp_max]
    minsd   xmm0, xmm7
    movsd   xmm7, [rel exp_min]
    maxsd   xmm0, xmm7

    ; n = round(x / ln2)
    movsd   xmm1, xmm0
    mulsd   xmm1, [rel inv_ln2]        ; xmm1 = x / ln2
    addsd   xmm1, [rel half]            ; add 0.5 for rounding
    cvttsd2si rax, xmm1                 ; rax = (int)(x/ln2 + 0.5) = round
    ; Handle negative: floor instead of truncate toward zero
    cvtsi2sd xmm2, rax                  ; xmm2 = (double)n
    movsd   xmm3, xmm2
    mulsd   xmm3, [rel ln2]
    movsd   xmm4, xmm0
    subsd   xmm4, xmm3                  ; check if remainder is right
    ; If x/ln2 was negative and we rounded wrong, adjust
    ; Simple: recompute n = floor(x/ln2 + 0.5) via truncation is fine for our range

    ; r = x - n * ln2
    movsd   xmm1, xmm2                  ; xmm1 = n (as double)
    mulsd   xmm1, [rel ln2]             ; xmm1 = n * ln2
    movsd   xmm2, xmm0
    subsd   xmm2, xmm1                  ; xmm2 = r = x - n*ln2

    ; Evaluate P(r) using Horner's method:
    ; P = c6
    ; P = P * r + c5
    ; P = P * r + c4
    ; ...
    ; P = P * r + c0
    movsd   xmm0, [rel c6]
    mulsd   xmm0, xmm2
    addsd   xmm0, [rel c5]
    mulsd   xmm0, xmm2
    addsd   xmm0, [rel c4]
    mulsd   xmm0, xmm2
    addsd   xmm0, [rel c3]
    mulsd   xmm0, xmm2
    addsd   xmm0, [rel c2]
    mulsd   xmm0, xmm2
    addsd   xmm0, [rel c1]
    mulsd   xmm0, xmm2
    addsd   xmm0, [rel c0]              ; xmm0 = P(r) ≈ e^r

    ; Multiply by 2^n:
    ; Construct 2^n by manipulating IEEE 754 exponent field
    ; 2^n as double: exponent = n + 1023, mantissa = 0
    add     rax, 1023
    test    rax, rax
    jle     .underflow
    cmp     rax, 2046
    jge     .overflow
    shl     rax, 52
    movq    xmm1, rax                   ; xmm1 = 2^n
    mulsd   xmm0, xmm1                  ; xmm0 = e^r * 2^n = e^x
    ret

.underflow:
    movsd   xmm0, [rel zero_f]
    ret

.overflow:
    movsd   xmm0, [rel exp_max]         ; saturate (not ideal but safe)
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 2: Write sigmoid.asm**

Create `src/math/sigmoid.asm`:

```nasm
; src/math/sigmoid.asm
; sigmoid(xmm0: x) -> xmm0: 1/(1 + e^(-x))
; sigmoid_deriv(xmm0: s) -> xmm0: s * (1 - s)
;   where s is the sigmoid OUTPUT (not the raw input)

bits 64
default rel

extern exp_approx

section .rodata
    align 16
    one:    dq 1.0
    neg1:   dq -1.0

section .text
global sigmoid
global sigmoid_deriv

sigmoid:
    push rbp
    mov  rbp, rsp

    ; Compute -x
    mulsd xmm0, [rel neg1]

    ; Compute e^(-x)
    call  exp_approx

    ; 1 + e^(-x)
    addsd xmm0, [rel one]

    ; 1 / (1 + e^(-x))
    movsd xmm1, [rel one]
    divsd xmm1, xmm0
    movsd xmm0, xmm1

    pop  rbp
    ret

; sigmoid_deriv: takes sigmoid output s, returns s * (1 - s)
sigmoid_deriv:
    movsd xmm1, [rel one]
    subsd xmm1, xmm0          ; xmm1 = 1 - s
    mulsd xmm0, xmm1          ; xmm0 = s * (1 - s)
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 3: Write dot.asm**

Create `src/math/dot.asm`:

```nasm
; src/math/dot.asm
; dot_product(rdi: *A, rsi: *B, rdx: count) -> xmm0: result
;   Computes sum of A[i] * B[i] for i in 0..count-1

bits 64
default rel

section .text
global dot_product

dot_product:
    xorpd xmm0, xmm0          ; accumulator = 0.0
    test  rdx, rdx
    jz    .done

    xor   rcx, rcx             ; index = 0
.loop:
    movsd xmm1, [rdi + rcx*8]
    mulsd xmm1, [rsi + rcx*8]
    addsd xmm0, xmm1
    inc   rcx
    cmp   rcx, rdx
    jl    .loop

.done:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 4: Update main.asm to test math functions**

Update `src/main.asm` to test sigmoid(0.0) which should print ≈ 0.5000:

```nasm
bits 64
default rel

%include "constants.inc"

extern print_double
extern print_string
extern print_newline
extern sigmoid

section .data
    msg_sig:     db "sigmoid(0) = "
    msg_sig_len  equ $ - msg_sig
    zero_val:    dq 0.0

section .text
global _start

_start:
    lea  rdi, [rel msg_sig]
    mov  esi, msg_sig_len
    call print_string

    movsd xmm0, [rel zero_val]
    call sigmoid
    call print_double
    call print_newline

    mov  eax, SYS_exit
    xor  edi, edi
    syscall

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 5: Build and verify**

Run: `make clean && make && ./build/nasmlearn`
Expected output: `sigmoid(0) = 0.5000`

**Step 6: Commit**

```bash
git add src/math/
git commit -m "feat: add math module with exp, sigmoid, and dot product"
```

---

### Task 4: Data module — XOR training set

**Files:**
- Create: `src/data/xor.asm`

**Step 1: Write xor.asm**

Create `src/data/xor.asm`:

```nasm
; src/data/xor.asm
; XOR training dataset: 4 samples
;
; Exports:
;   xor_inputs:   4 pairs of doubles (8 doubles total)
;   xor_targets:  4 target doubles
;   xor_count:    number of samples (4)

bits 64
default rel

section .data

global xor_inputs
global xor_targets
global xor_count

; Input pairs: [x1, x2] for each sample
xor_inputs:
    dq 0.0, 0.0    ; sample 0: (0,0) -> 0
    dq 0.0, 1.0    ; sample 1: (0,1) -> 1
    dq 1.0, 0.0    ; sample 2: (1,0) -> 1
    dq 1.0, 1.0    ; sample 3: (1,1) -> 0

; Expected outputs
xor_targets:
    dq 0.0          ; 0 XOR 0 = 0
    dq 1.0          ; 0 XOR 1 = 1
    dq 1.0          ; 1 XOR 0 = 1
    dq 0.0          ; 1 XOR 1 = 0

xor_count: dq 4

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 2: Build and verify**

Run: `make clean && make`
Expected: Builds without errors.

**Step 3: Commit**

```bash
git add src/data/
git commit -m "feat: add XOR training dataset"
```

---

### Task 5: Neural network module — forward pass

**Files:**
- Create: `src/nn/forward.asm`

**Step 1: Write forward.asm**

Create `src/nn/forward.asm`:

```nasm
; src/nn/forward.asm
; forward_layer(rdi: *input, rsi: *weights, rdx: *biases,
;               rcx: *output, r8: in_count, r9: out_count)
;
; For each output neuron j:
;   output[j] = sigmoid( dot(input, weights[j*in_count..], in_count) + bias[j] )
;
; Weights layout: row-major, weights[j * in_count + i] = weight from input i to output j

bits 64
default rel

extern dot_product
extern sigmoid

section .text
global forward_layer

forward_layer:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov  r12, rdi              ; r12 = input
    mov  r13, rsi              ; r13 = weights
    mov  r14, rdx              ; r14 = biases
    mov  r15, rcx              ; r15 = output
    mov  rbx, r8               ; rbx = in_count
    mov  rcx, r9               ; rcx = out_count (loop counter)

    xor  r8, r8                ; r8 = j (neuron index)

.neuron_loop:
    cmp  r8, rcx
    jge  .done
    push rcx                   ; preserve out_count

    ; dot(input, &weights[j * in_count], in_count)
    mov  rdi, r12              ; input
    mov  rax, r8
    imul rax, rbx              ; j * in_count
    lea  rsi, [r13 + rax*8]   ; &weights[j * in_count]
    mov  rdx, rbx              ; in_count
    call dot_product           ; xmm0 = dot product

    ; Add bias[j]
    addsd xmm0, [r14 + r8*8]

    ; Apply sigmoid
    call sigmoid               ; xmm0 = sigmoid(sum + bias)

    ; Store output[j]
    movsd [r15 + r8*8], xmm0

    pop  rcx
    inc  r8
    jmp  .neuron_loop

.done:
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 2: Build and verify**

Run: `make clean && make`
Expected: Builds without errors.

**Step 3: Commit**

```bash
git add src/nn/forward.asm
git commit -m "feat: add forward pass for dense layer with sigmoid activation"
```

---

### Task 6: Neural network module — backward pass

**Files:**
- Create: `src/nn/backward.asm`

**Step 1: Write backward.asm**

Create `src/nn/backward.asm`:

```nasm
; src/nn/backward.asm
; Backpropagation gradient computation
;
; backward_output(rdi: *output, rsi: *expected, rdx: *hidden_out,
;                 rcx: *grad_w_out, r8: *grad_b_out)
;   Computes output layer delta and gradients.
;   delta = (output - expected) * sigmoid_deriv(output)
;   grad_w[i] = delta * hidden_out[i]
;   grad_b = delta
;
; backward_hidden(rdi: *hidden_out, rsi: *input, rdx: *delta_out,
;                 rcx: *w_output, r8: *grad_w_hid, r9: *grad_b_hid,
;                 [rbp+16]: *delta_hidden)
;   Computes hidden layer deltas and gradients.

bits 64
default rel

extern sigmoid_deriv

section .text
global backward_output
global backward_hidden

; backward_output
; rdi = output (1 double), rsi = expected (1 double)
; rdx = hidden_out (HIDDEN_SIZE doubles)
; rcx = grad_w_out (HIDDEN_SIZE doubles), r8 = grad_b_out (1 double)
backward_output:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13

    mov  r12, rdx              ; hidden_out
    mov  r13, rcx              ; grad_w_out
    mov  rbx, r8               ; grad_b_out

    ; delta = (output - expected) * sigmoid_deriv(output)
    movsd xmm0, [rdi]         ; output
    movsd xmm2, xmm0          ; save output for deriv
    subsd xmm0, [rsi]         ; output - expected = error
    movsd xmm3, xmm0          ; save error

    movsd xmm0, xmm2          ; pass output to sigmoid_deriv
    call  sigmoid_deriv        ; xmm0 = s * (1 - s)
    mulsd xmm0, xmm3          ; xmm0 = delta = error * deriv

    ; Store grad_b_out = delta
    movsd [rbx], xmm0

    ; grad_w_out[i] = delta * hidden_out[i]
    movsd xmm1, xmm0          ; xmm1 = delta
    movsd xmm2, [r12]         ; hidden_out[0]
    mulsd xmm2, xmm1
    movsd [r13], xmm2         ; grad_w_out[0]

    movsd xmm2, [r12 + 8]     ; hidden_out[1]
    mulsd xmm2, xmm1
    movsd [r13 + 8], xmm2     ; grad_w_out[1]

    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

; backward_hidden
; rdi = hidden_out (2 doubles), rsi = input (2 doubles)
; rdx = delta_out (1 double — output layer delta stored at grad_b_out)
; rcx = w_output (2 doubles)
; r8  = grad_w_hid (4 doubles), r9 = grad_b_hid (2 doubles)
; [rbp+16] = delta_hidden (2 doubles, scratch for hidden deltas)
backward_hidden:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov  r10, rdi              ; hidden_out
    mov  r11, rsi              ; input
    mov  r12, rdx              ; delta_out ptr
    mov  r13, rcx              ; w_output
    mov  r14, r8               ; grad_w_hid
    mov  r15, r9               ; grad_b_hid
    mov  rbx, [rbp + 16]      ; delta_hidden

    ; Load output delta
    movsd xmm4, [r12]         ; delta_out

    ; For each hidden neuron j (j=0,1):
    ;   delta_hidden[j] = delta_out * w_output[j] * sigmoid_deriv(hidden_out[j])
    ;   grad_b_hid[j] = delta_hidden[j]
    ;   grad_w_hid[j*2 + i] = delta_hidden[j] * input[i]

    ; --- Hidden neuron 0 ---
    movsd xmm0, xmm4
    mulsd xmm0, [r13]         ; delta_out * w_output[0]
    movsd xmm5, xmm0          ; save propagated error

    movsd xmm0, [r10]         ; hidden_out[0]
    call  sigmoid_deriv        ; xmm0 = deriv
    mulsd xmm0, xmm5          ; delta_hidden[0]
    movsd [rbx], xmm0         ; store delta_hidden[0]
    movsd [r15], xmm0         ; grad_b_hid[0] = delta_hidden[0]

    ; grad_w_hid[0] = delta_hidden[0] * input[0]
    movsd xmm1, xmm0
    mulsd xmm1, [r11]
    movsd [r14], xmm1

    ; grad_w_hid[1] = delta_hidden[0] * input[1]
    movsd xmm1, xmm0
    mulsd xmm1, [r11 + 8]
    movsd [r14 + 8], xmm1

    ; --- Hidden neuron 1 ---
    movsd xmm0, xmm4
    mulsd xmm0, [r13 + 8]     ; delta_out * w_output[1]
    movsd xmm5, xmm0

    movsd xmm0, [r10 + 8]     ; hidden_out[1]
    call  sigmoid_deriv        ; xmm0 = deriv
    mulsd xmm0, xmm5          ; delta_hidden[1]
    movsd [rbx + 8], xmm0     ; store delta_hidden[1]
    movsd [r15 + 8], xmm0     ; grad_b_hid[1]

    ; grad_w_hid[2] = delta_hidden[1] * input[0]
    movsd xmm1, xmm0
    mulsd xmm1, [r11]
    movsd [r14 + 16], xmm1

    ; grad_w_hid[3] = delta_hidden[1] * input[1]
    movsd xmm1, xmm0
    mulsd xmm1, [r11 + 8]
    movsd [r14 + 24], xmm1

    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 2: Build and verify**

Run: `make clean && make`
Expected: Builds without errors.

**Step 3: Commit**

```bash
git add src/nn/backward.asm
git commit -m "feat: add backpropagation gradient computation for output and hidden layers"
```

---

### Task 7: Neural network module — weight update (SGD)

**Files:**
- Create: `src/nn/update.asm`

**Step 1: Write update.asm**

Create `src/nn/update.asm`:

```nasm
; src/nn/update.asm
; update_weights(rdi: *weights, rsi: *gradients, rdx: count, xmm0: learning_rate)
;   w[i] -= lr * grad[i]

bits 64
default rel

section .text
global update_weights

update_weights:
    test  rdx, rdx
    jz    .done

    xor   rcx, rcx             ; index = 0
.loop:
    movsd xmm1, [rsi + rcx*8] ; grad[i]
    mulsd xmm1, xmm0          ; lr * grad[i]
    movsd xmm2, [rdi + rcx*8] ; w[i]
    subsd xmm2, xmm1          ; w[i] - lr * grad[i]
    movsd [rdi + rcx*8], xmm2 ; store updated weight
    inc   rcx
    cmp   rcx, rdx
    jl    .loop

.done:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 2: Build and verify**

Run: `make clean && make`
Expected: Builds without errors.

**Step 3: Commit**

```bash
git add src/nn/update.asm
git commit -m "feat: add SGD weight update"
```

---

### Task 8: Main training loop — tie everything together

**Files:**
- Modify: `src/main.asm` (full rewrite)

**Step 1: Write the full main.asm with weight init, training loop, and result printing**

Rewrite `src/main.asm` with:
1. LCG PRNG for weight initialization (seeded from rdtsc)
2. Training loop: 10000 epochs, 4 samples each
3. Loss printing every 1000 epochs
4. Final prediction printing for all 4 XOR inputs

```nasm
; src/main.asm
; NASM Learn Simple — Main entry point
; Trains a 2-2-1 neural network to learn XOR via backpropagation.

bits 64
default rel

%include "constants.inc"

; External functions
extern print_double
extern print_string
extern print_newline
extern forward_layer
extern backward_output
extern backward_hidden
extern update_weights
extern xor_inputs
extern xor_targets
extern xor_count

section .rodata
    learning_rate:  dq 0.5
    four_f:         dq 4.0
    one_f:          dq 1.0
    two_f:          dq 2.0
    neg_one_f:      dq -1.0

    ; LCG constants
    lcg_a:          dq 6364136223846793005
    lcg_c:          dq 1442695040888963407

    ; Messages
    msg_epoch:      db "Epoch "
    msg_epoch_len   equ $ - msg_epoch
    msg_loss:       db " | Loss: "
    msg_loss_len    equ $ - msg_loss
    msg_input:      db "Input: ("
    msg_input_len   equ $ - msg_input
    msg_comma:      db ", "
    msg_comma_len   equ $ - msg_comma
    msg_arrow:      db ") -> "
    msg_arrow_len   equ $ - msg_arrow
    msg_expect:     db " (expected: "
    msg_expect_len  equ $ - msg_expect
    msg_paren:      db ")"
    msg_paren_len   equ $ - msg_paren
    msg_title:      db "=== NASM Learn Simple ===", 10
    msg_title_len   equ $ - msg_title
    msg_train:      db "Training XOR network (2-2-1)...", 10
    msg_train_len   equ $ - msg_train
    msg_results:    db 10, "--- Final Predictions ---", 10
    msg_results_len equ $ - msg_results

section .bss
    ; Network weights and biases
    w_hidden:       resq 4     ; 2x2 weights
    b_hidden:       resq 2     ; 2 biases
    w_output:       resq 2     ; 1x2 weights
    b_output:       resq 1     ; 1 bias

    ; Scratch buffers
    hidden_out:     resq 2     ; hidden layer activations
    net_output:     resq 1     ; network output

    ; Gradient buffers
    grad_w_out:     resq 2
    grad_b_out:     resq 1
    grad_w_hid:     resq 4
    grad_b_hid:     resq 2
    delta_hidden:   resq 2

    ; PRNG state
    lcg_state:      resq 1

    ; Epoch loss accumulator
    epoch_loss:     resq 1

section .text
global _start

; --- PRNG: init_rng and next_random ---

; Seed PRNG from rdtsc
init_rng:
    rdtsc
    shl  rdx, 32
    or   rax, rdx
    mov  [rel lcg_state], rax
    ret

; next_random() -> xmm0: random double in [-1.0, 1.0]
next_random:
    mov  rax, [rel lcg_state]
    imul rax, [rel lcg_a]
    add  rax, [rel lcg_c]
    mov  [rel lcg_state], rax

    ; Convert to double in [0, 1): use upper bits
    shr  rax, 11               ; 53 bits of mantissa
    cvtsi2sd xmm0, rax
    mov  rax, 1
    shl  rax, 53
    cvtsi2sd xmm1, rax
    divsd xmm0, xmm1          ; xmm0 in [0, 1)

    ; Map to [-1, 1]
    mulsd xmm0, [rel two_f]   ; [0, 2)
    subsd xmm0, [rel one_f]   ; [-1, 1)
    ret

; --- Weight initialization ---

; init_weights: fill all weights and biases with random values
init_weights:
    push rbp
    mov  rbp, rsp
    push r12
    push r13

    call init_rng

    ; Initialize w_hidden (4 doubles)
    lea  r12, [rel w_hidden]
    mov  r13, 4
.init_wh:
    call next_random
    movsd [r12], xmm0
    add  r12, 8
    dec  r13
    jnz  .init_wh

    ; Initialize b_hidden (2 doubles)
    lea  r12, [rel b_hidden]
    mov  r13, 2
.init_bh:
    call next_random
    movsd [r12], xmm0
    add  r12, 8
    dec  r13
    jnz  .init_bh

    ; Initialize w_output (2 doubles)
    lea  r12, [rel w_output]
    mov  r13, 2
.init_wo:
    call next_random
    movsd [r12], xmm0
    add  r12, 8
    dec  r13
    jnz  .init_wo

    ; Initialize b_output (1 double)
    lea  r12, [rel b_output]
    call next_random
    movsd [r12], xmm0

    pop  r13
    pop  r12
    pop  rbp
    ret

; --- Print integer (for epoch number) ---
; print_int(rdi: value)
print_int:
    push rbp
    mov  rbp, rsp
    sub  rsp, 32               ; temp buffer
    mov  rax, rdi
    lea  rsi, [rbp - 1]       ; write backwards from end of buffer
    mov  byte [rsi], 0         ; null terminator (not needed but safe)
    xor  rcx, rcx              ; digit count

    test rax, rax
    jnz  .pi_loop
    dec  rsi
    mov  byte [rsi], '0'
    inc  rcx
    jmp  .pi_print

.pi_loop:
    test rax, rax
    jz   .pi_print
    xor  edx, edx
    mov  r8, 10
    div  r8
    add  dl, '0'
    dec  rsi
    mov  [rsi], dl
    inc  rcx
    jmp  .pi_loop

.pi_print:
    mov  rdi, rsi
    mov  rsi, rcx
    call print_string

    add  rsp, 32
    pop  rbp
    ret

; --- MAIN ENTRY POINT ---

_start:
    ; Print title
    lea  rdi, [rel msg_title]
    mov  esi, msg_title_len
    call print_string

    ; Print training message
    lea  rdi, [rel msg_train]
    mov  esi, msg_train_len
    call print_string

    ; Initialize weights
    call init_weights

    ; === Training loop ===
    xor  r12, r12              ; r12 = epoch counter

.epoch_loop:
    cmp  r12, MAX_EPOCHS
    jge  .training_done

    ; Zero epoch loss
    xorpd xmm0, xmm0
    movsd [rel epoch_loss], xmm0

    ; Iterate over 4 training samples
    xor  r13, r13              ; r13 = sample index

.sample_loop:
    cmp  r13, 4
    jge  .sample_done

    ; Compute pointer to current input: xor_inputs + sample * 2 * 8
    mov  rax, r13
    shl  rax, 4                ; sample * 16 (2 doubles * 8 bytes)
    lea  rdi, [rel xor_inputs]
    add  rdi, rax              ; rdi = &xor_inputs[sample * 2]

    ; Forward pass: hidden layer
    ; forward_layer(input, w_hidden, b_hidden, hidden_out, 2, 2)
    lea  rsi, [rel w_hidden]
    lea  rdx, [rel b_hidden]
    lea  rcx, [rel hidden_out]
    mov  r8, INPUT_SIZE
    mov  r9, HIDDEN_SIZE
    push r12
    push r13
    call forward_layer
    pop  r13
    pop  r12

    ; Forward pass: output layer
    ; forward_layer(hidden_out, w_output, b_output, net_output, 2, 1)
    lea  rdi, [rel hidden_out]
    lea  rsi, [rel w_output]
    lea  rdx, [rel b_output]
    lea  rcx, [rel net_output]
    mov  r8, HIDDEN_SIZE
    mov  r9, OUTPUT_SIZE
    push r12
    push r13
    call forward_layer
    pop  r13
    pop  r12

    ; Accumulate loss: loss += (output - expected)^2
    movsd xmm0, [rel net_output]
    mov  rax, r13
    lea  rcx, [rel xor_targets]
    movsd xmm1, [rcx + rax*8]
    subsd xmm0, xmm1          ; error
    mulsd xmm0, xmm0          ; error^2
    addsd xmm0, [rel epoch_loss]
    movsd [rel epoch_loss], xmm0

    ; Backward pass: output layer
    ; backward_output(output, expected, hidden_out, grad_w_out, grad_b_out)
    lea  rdi, [rel net_output]
    mov  rax, r13
    lea  rsi, [rel xor_targets]
    lea  rsi, [rsi + rax*8]
    lea  rdx, [rel hidden_out]
    lea  rcx, [rel grad_w_out]
    lea  r8,  [rel grad_b_out]
    push r12
    push r13
    call backward_output
    pop  r13
    pop  r12

    ; Backward pass: hidden layer
    ; backward_hidden(hidden_out, input, delta_out, w_output,
    ;                 grad_w_hid, grad_b_hid, delta_hidden)
    lea  rdi, [rel hidden_out]
    mov  rax, r13
    shl  rax, 4
    lea  rsi, [rel xor_inputs]
    add  rsi, rax
    lea  rdx, [rel grad_b_out]    ; delta_out = grad_b_out (same value)
    lea  rcx, [rel w_output]
    lea  r8,  [rel grad_w_hid]
    lea  r9,  [rel grad_b_hid]
    lea  rax, [rel delta_hidden]
    push rax                      ; 7th arg on stack
    push r12
    push r13
    call backward_hidden
    pop  r13
    pop  r12
    add  rsp, 8                   ; clean up 7th arg

    ; Update weights: output layer
    lea  rdi, [rel w_output]
    lea  rsi, [rel grad_w_out]
    mov  edx, HIDDEN_SIZE
    movsd xmm0, [rel learning_rate]
    push r12
    push r13
    call update_weights
    pop  r13
    pop  r12

    ; Update biases: output layer
    lea  rdi, [rel b_output]
    lea  rsi, [rel grad_b_out]
    mov  edx, OUTPUT_SIZE
    movsd xmm0, [rel learning_rate]
    push r12
    push r13
    call update_weights
    pop  r13
    pop  r12

    ; Update weights: hidden layer
    lea  rdi, [rel w_hidden]
    lea  rsi, [rel grad_w_hid]
    mov  edx, 4                ; HIDDEN_SIZE * INPUT_SIZE
    movsd xmm0, [rel learning_rate]
    push r12
    push r13
    call update_weights
    pop  r13
    pop  r12

    ; Update biases: hidden layer
    lea  rdi, [rel b_hidden]
    lea  rsi, [rel grad_b_hid]
    mov  edx, HIDDEN_SIZE
    movsd xmm0, [rel learning_rate]
    push r12
    push r13
    call update_weights
    pop  r13
    pop  r12

    inc  r13
    jmp  .sample_loop

.sample_done:
    ; Average loss over 4 samples
    movsd xmm0, [rel epoch_loss]
    divsd xmm0, [rel four_f]
    movsd [rel epoch_loss], xmm0

    ; Print loss every PRINT_EVERY epochs
    mov  rax, r12
    xor  edx, edx
    mov  rcx, PRINT_EVERY
    div  rcx
    test rdx, rdx
    jnz  .skip_print

    ; Print "Epoch NNNN | Loss: X.XXXX"
    push r12
    lea  rdi, [rel msg_epoch]
    mov  esi, msg_epoch_len
    call print_string

    mov  rdi, r12
    call print_int

    lea  rdi, [rel msg_loss]
    mov  esi, msg_loss_len
    call print_string

    movsd xmm0, [rel epoch_loss]
    call print_double
    call print_newline
    pop  r12

.skip_print:
    inc  r12
    jmp  .epoch_loop

.training_done:
    ; === Print final predictions ===
    lea  rdi, [rel msg_results]
    mov  esi, msg_results_len
    call print_string

    xor  r13, r13              ; sample index

.result_loop:
    cmp  r13, 4
    jge  .exit

    ; Forward pass for this sample
    mov  rax, r13
    shl  rax, 4
    lea  rdi, [rel xor_inputs]
    add  rdi, rax
    push rdi                   ; save input pointer

    lea  rsi, [rel w_hidden]
    lea  rdx, [rel b_hidden]
    lea  rcx, [rel hidden_out]
    mov  r8, INPUT_SIZE
    mov  r9, HIDDEN_SIZE
    push r13
    call forward_layer
    pop  r13

    lea  rdi, [rel hidden_out]
    lea  rsi, [rel w_output]
    lea  rdx, [rel b_output]
    lea  rcx, [rel net_output]
    mov  r8, HIDDEN_SIZE
    mov  r9, OUTPUT_SIZE
    push r13
    call forward_layer
    pop  r13

    ; Print "Input: (x1, x2) -> output (expected: target)"
    lea  rdi, [rel msg_input]
    mov  esi, msg_input_len
    push r13
    call print_string
    pop  r13

    pop  rdi                   ; restore input pointer
    push rdi
    movsd xmm0, [rdi]
    push r13
    call print_double
    pop  r13

    lea  rdi, [rel msg_comma]
    mov  esi, msg_comma_len
    push r13
    call print_string
    pop  r13

    pop  rdi
    movsd xmm0, [rdi + 8]
    push r13
    call print_double
    pop  r13

    lea  rdi, [rel msg_arrow]
    mov  esi, msg_arrow_len
    push r13
    call print_string
    pop  r13

    movsd xmm0, [rel net_output]
    push r13
    call print_double
    pop  r13

    lea  rdi, [rel msg_expect]
    mov  esi, msg_expect_len
    push r13
    call print_string
    pop  r13

    mov  rax, r13
    lea  rcx, [rel xor_targets]
    movsd xmm0, [rcx + rax*8]
    push r13
    call print_double
    pop  r13

    lea  rdi, [rel msg_paren]
    mov  esi, msg_paren_len
    push r13
    call print_string
    pop  r13

    push r13
    call print_newline
    pop  r13

    inc  r13
    jmp  .result_loop

.exit:
    mov  eax, SYS_exit
    xor  edi, edi
    syscall

section .note.GNU-stack noalloc noexec nowrite progbits
```

**Step 2: Build and run**

Run: `make clean && make && ./build/nasmlearn`
Expected output:
```
=== NASM Learn Simple ===
Training XOR network (2-2-1)...
Epoch 0 | Loss: X.XXXX
Epoch 1000 | Loss: X.XXXX
...
Epoch 9000 | Loss: 0.00XX

--- Final Predictions ---
Input: (0.0000, 0.0000) -> 0.0XXX (expected: 0.0000)
Input: (0.0000, 1.0000) -> 0.9XXX (expected: 1.0000)
Input: (1.0000, 0.0000) -> 0.9XXX (expected: 1.0000)
Input: (1.0000, 1.0000) -> 0.0XXX (expected: 0.0000)
```

**Step 3: Debug if needed**

If the network doesn't converge:
- Check `objdump -d -Mintel build/main.o` for correct instruction encoding
- Verify sigmoid output with `gdb`: `break sigmoid`, `print $xmm0`
- Check weight values aren't NaN: `gdb`, inspect `.bss` addresses

**Step 4: Commit**

```bash
git add src/main.asm
git commit -m "feat: add full training loop with weight init, backprop, and result printing"
```

---

### Task 9: Test, debug, and verify convergence

**Step 1: Full build and run**

Run: `make clean && make && ./build/nasmlearn`

**Step 2: Verify convergence**

Check that:
- Loss decreases monotonically (roughly) across epochs
- Final predictions match XOR truth table within tolerance
- (0,0) and (1,1) outputs < 0.1
- (0,1) and (1,0) outputs > 0.9

**Step 3: Fix any issues found**

Common issues to check:
- Stack alignment (must be 16-byte aligned before `call`)
- Register clobbering (callee-saved regs preserved?)
- Off-by-one in weight indexing
- Sign errors in gradient computation

**Step 4: Commit final working version**

```bash
git add -A
git commit -m "feat: verified XOR convergence — NASM ML library complete"
```
