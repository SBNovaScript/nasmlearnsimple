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
    sub  rsp, 24              ; 16 bytes for locals + 8 for alignment
                               ; Stack: 6 pushes (48 bytes) + 24 = 72 from entry
                               ; Entry RSP%16==8, so 8+72=80, 80%16==0. Aligned!

    ; Save args to callee-saved regs and stack locals
    mov  r12, rdi              ; input ptr
    mov  r13, rsi              ; weights ptr
    mov  r14, rdx              ; biases ptr
    mov  r15, rcx              ; output ptr
    mov  rbx, r8               ; in_count
    mov  [rbp-48], r9          ; out_count (local variable)
    mov  qword [rbp-56], 0     ; j = 0 (neuron index)

.neuron_loop:
    mov  rax, [rbp-56]         ; j
    cmp  rax, [rbp-48]         ; j < out_count?
    jge  .done

    ; dot_product(input, &weights[j * in_count], in_count)
    mov  rdi, r12              ; input
    mov  rax, [rbp-56]         ; j
    imul rax, rbx              ; j * in_count
    lea  rsi, [r13 + rax*8]   ; &weights[j * in_count]
    mov  rdx, rbx              ; in_count
    call dot_product           ; xmm0 = dot product result

    ; Add bias[j]
    mov  rax, [rbp-56]
    addsd xmm0, [r14 + rax*8]

    ; sigmoid(sum)
    call sigmoid               ; xmm0 = sigmoid(dot + bias)

    ; Store output[j]
    mov  rax, [rbp-56]
    movsd [r15 + rax*8], xmm0

    ; j++
    inc  qword [rbp-56]
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
