; src/math/dot.asm
; Dot product of two double arrays

bits 64
default rel

section .text
global dot_product

; dot_product(rdi: *A, rsi: *B, rdx: count) -> xmm0: result
; Leaf function.
dot_product:
    xorpd   xmm0, xmm0
    xor     ecx, ecx
    test    rdx, rdx
    jz      .done

.loop:
    movsd   xmm1, [rdi + rcx*8]
    mulsd   xmm1, [rsi + rcx*8]
    addsd   xmm0, xmm1
    inc     rcx
    cmp     rcx, rdx
    jb      .loop

.done:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
