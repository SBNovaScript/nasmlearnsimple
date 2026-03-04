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

    xor   ecx, ecx
.loop:
    movsd xmm1, [rsi + rcx*8]
    mulsd xmm1, xmm0
    movsd xmm2, [rdi + rcx*8]
    subsd xmm2, xmm1
    movsd [rdi + rcx*8], xmm2
    inc   rcx
    cmp   rcx, rdx
    jl    .loop

.done:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
