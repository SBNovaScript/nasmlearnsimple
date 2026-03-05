; src/nn/forward.asm
; forward_layer(rdi: *input, rsi: *weights, rdx: *biases,
;               rcx: *output, r8: in_count, r9: out_count)

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
    sub  rsp, 24              ; 2 locals + alignment
                               ; 6 pushes (48) + 24 = 72. Entry RSP%16==8.
                               ; 8+72=80, 80%16==0. Aligned.

    mov  r12, rdi              ; input ptr
    mov  r13, rsi              ; weights ptr
    mov  r14, rdx              ; biases ptr
    mov  r15, rcx              ; output ptr
    mov  [rbp-48], r8          ; in_count
    mov  [rbp-56], r9          ; out_count
    xor  ebx, ebx              ; j = 0 (neuron index)

.neuron_loop:
    cmp  rbx, [rbp-56]         ; j < out_count?
    jge  .done

    ; dot_product(input, &weights[j * in_count], in_count)
    mov  rdi, r12
    mov  rax, rbx
    imul rax, [rbp-48]         ; j * in_count
    lea  rsi, [r13 + rax*8]
    mov  rdx, [rbp-48]
    call dot_product

    addsd xmm0, [r14 + rbx*8] ; + bias[j]
    call sigmoid

    movsd [r15 + rbx*8], xmm0 ; output[j] = sigmoid(dot + bias)
    inc  rbx
    jmp  .neuron_loop

.done:
    add  rsp, 24
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
