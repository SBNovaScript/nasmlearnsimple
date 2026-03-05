; SPDX-License-Identifier: Apache-2.0
; Copyright (c) 2026-present Steven Baumann
; Original repository: https://github.com/SBNovaScript/nasmlearnsimple
; See LICENSE and NOTICE in the repository root for details.
;
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
; Calls exp_approx.
sigmoid:
    push    rbp
    mov     rbp, rsp
    ; Entry RSP%16==8, after push: RSP%16==0. Aligned.

    mulsd   xmm0, [rel neg_one]     ; -x
    call    exp_approx               ; e^(-x)

    addsd   xmm0, [rel pos_one]     ; 1 + e^(-x)
    movsd   xmm1, [rel pos_one]
    divsd   xmm1, xmm0              ; 1 / (1 + e^(-x))
    movsd   xmm0, xmm1

    pop     rbp
    ret

; sigmoid_deriv(xmm0: s) -> xmm0: s * (1 - s)
; Takes sigmoid OUTPUT, not raw x. Leaf function.
sigmoid_deriv:
    movsd   xmm1, [rel pos_one]
    subsd   xmm1, xmm0
    mulsd   xmm0, xmm1
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
