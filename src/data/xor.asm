; src/data/xor.asm
; XOR training dataset: 4 samples

bits 64
default rel

section .data

global xor_inputs
global xor_targets

xor_inputs:
    dq 0.0, 0.0
    dq 0.0, 1.0
    dq 1.0, 0.0
    dq 1.0, 1.0

xor_targets:
    dq 0.0
    dq 1.0
    dq 1.0
    dq 0.0

section .note.GNU-stack noalloc noexec nowrite progbits
