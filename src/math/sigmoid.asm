; src/math/sigmoid.asm
; Sigmoid function and its derivative

bits 64
default rel

extern exp_approx

section .rodata
    align 16
    neg_one:    dq -1.0
    pos_one:    dq  1.0

section .text
global sigmoid
global sigmoid_deriv

; sigmoid(xmm0: x) -> xmm0: 1 / (1 + e^(-x))
; Calls exp_approx, so needs frame pointer and proper stack alignment.
sigmoid:
    push    rbp
    mov     rbp, rsp
    ; At entry RSP % 16 == 8 (return addr pushed by call).
    ; After push rbp: RSP % 16 == 0.
    ; Next call will push return addr making callee see RSP % 16 == 8. Correct.

    ; Negate x: xmm0 = x * (-1.0)
    mulsd   xmm0, [rel neg_one]

    ; Call exp_approx(-x)
    call    exp_approx

    ; xmm0 = e^(-x)
    ; Compute 1.0 + e^(-x)
    addsd   xmm0, [rel pos_one]     ; xmm0 = 1 + e^(-x)

    ; Compute 1.0 / (1 + e^(-x))
    movsd   xmm1, [rel pos_one]     ; xmm1 = 1.0
    divsd   xmm1, xmm0             ; xmm1 = 1.0 / (1 + e^(-x))
    movsd   xmm0, xmm1             ; xmm0 = sigmoid(x)

    pop     rbp
    ret

; sigmoid_deriv(xmm0: s) -> xmm0: s * (1 - s)
; s is the sigmoid OUTPUT (not raw x). Leaf function.
sigmoid_deriv:
    movsd   xmm1, [rel pos_one]     ; xmm1 = 1.0
    subsd   xmm1, xmm0             ; xmm1 = 1.0 - s
    mulsd   xmm0, xmm1             ; xmm0 = s * (1 - s)
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
