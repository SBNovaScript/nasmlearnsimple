; SPDX-License-Identifier: Apache-2.0
; Copyright (c) 2026-present Steven Baumann
; Original repository: https://github.com/SBNovaScript/nasmlearnsimple
; See LICENSE and NOTICE in the repository root for details.
;
; src/math/exp.asm
; e^x approximation using range reduction + Horner's polynomial

bits 64
default rel

section .rodata
    align 16
    inv_ln2:    dq 1.4426950408889634    ; 1 / ln(2)
    ln2:        dq 0.6931471805599453    ; ln(2)
    one:        dq 1.0
    clamp_hi:   dq 709.0
    clamp_lo:   dq -709.0
    ; Taylor coefficients for e^r around r=0
    c6:         dq 0.0013888888888888889  ; 1/720
    c5:         dq 0.008333333333333333   ; 1/120
    c4:         dq 0.041666666666666664   ; 1/24
    c3:         dq 0.16666666666666666    ; 1/6
    c2:         dq 0.5
    ; c1 = 1.0, c0 = 1.0 (reuse 'one')

section .text
global exp_approx

; exp_approx(xmm0: x) -> xmm0: e^x
; Leaf function — no calls, no frame pointer needed.
exp_approx:
    ; Step 1: Clamp x to [-709, 709]
    maxsd   xmm0, [rel clamp_lo]
    minsd   xmm0, [rel clamp_hi]

    ; Step 2: n = round(x / ln2)
    movsd   xmm1, xmm0              ; save x
    mulsd   xmm0, [rel inv_ln2]
    roundsd xmm0, xmm0, 0           ; round to nearest even (SSE4.1)
    cvttsd2si rax, xmm0

    ; Step 3: r = x - n * ln2  (reduced argument, |r| <= ln2/2)
    cvtsi2sd xmm0, rax
    mulsd   xmm0, [rel ln2]
    subsd   xmm1, xmm0              ; xmm1 = r

    ; Check biased exponent for underflow/overflow
    mov     rcx, rax
    add     rcx, 1023
    cmp     rcx, 0
    jle     .underflow
    cmp     rcx, 2046
    jge     .overflow

    ; Step 4: P(r) via Horner's method (degree-6 Taylor)
    movsd   xmm0, [rel c6]
    mulsd   xmm0, xmm1
    addsd   xmm0, [rel c5]
    mulsd   xmm0, xmm1
    addsd   xmm0, [rel c4]
    mulsd   xmm0, xmm1
    addsd   xmm0, [rel c3]
    mulsd   xmm0, xmm1
    addsd   xmm0, [rel c2]
    mulsd   xmm0, xmm1
    addsd   xmm0, [rel one]
    mulsd   xmm0, xmm1
    addsd   xmm0, [rel one]

    ; Step 5: e^x = 2^n * P(r)  — construct 2^n via IEEE 754 exponent bits
    shl     rcx, 52
    movq    xmm1, rcx
    mulsd   xmm0, xmm1
    ret

.underflow:
    xorpd   xmm0, xmm0
    ret

.overflow:
    mov     rcx, 2046
    shl     rcx, 52
    mov     rdx, 0x000FFFFFFFFFFFFF
    or      rcx, rdx
    movq    xmm0, rcx
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
