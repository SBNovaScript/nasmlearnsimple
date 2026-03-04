; src/math/dot.asm
; Dot product of two double arrays

bits 64
default rel

section .text
global dot_product

; dot_product(rdi: *A, rsi: *B, rdx: count) -> xmm0: result
; Computes sum of A[i] * B[i] for i = 0..count-1.
; Leaf function — no calls, no frame pointer needed.
dot_product:
    xorpd   xmm0, xmm0              ; xmm0 = accumulator = 0.0
    xor     ecx, ecx                 ; rcx = index = 0
    test    rdx, rdx                 ; check if count == 0
    jz      .done

.loop:
    movsd   xmm1, [rdi + rcx*8]     ; xmm1 = A[i]
    mulsd   xmm1, [rsi + rcx*8]     ; xmm1 = A[i] * B[i]
    addsd   xmm0, xmm1              ; accumulator += A[i] * B[i]
    inc     rcx
    cmp     rcx, rdx
    jb      .loop

.done:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
