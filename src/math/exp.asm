; src/math/exp.asm
; e^x approximation using range reduction + Horner's polynomial

bits 64
default rel

section .rodata
    align 16
    inv_ln2:    dq 1.4426950408889634    ; 1 / ln(2)
    ln2:        dq 0.6931471805599453    ; ln(2)
    half:       dq 0.5
    one:        dq 1.0
    zero:       dq 0.0
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

    ; Step 2: Compute n = round(x / ln2)
    ; n = trunc(x * (1/ln2) + 0.5)  for x >= 0
    ; n = trunc(x * (1/ln2) - 0.5)  for x < 0
    movsd   xmm1, xmm0              ; xmm1 = x (save)
    mulsd   xmm0, [rel inv_ln2]     ; xmm0 = x / ln2
    ; Add or subtract 0.5 depending on sign for correct rounding
    movsd   xmm2, xmm0              ; xmm2 = x / ln2
    movsd   xmm3, [rel half]
    ; If xmm2 < 0, we need to subtract 0.5 instead of adding
    xorpd   xmm4, xmm4
    cmpltsd xmm4, xmm2              ; xmm4 = (0 < xmm2) ? all 1s : 0
    movsd   xmm5, xmm3              ; xmm5 = 0.5
    movsd   xmm6, xmm3              ; xmm6 = 0.5
    ; Negate xmm6 for the negative case
    movsd   xmm7, [rel zero]
    subsd   xmm7, xmm6              ; xmm7 = -0.5
    ; Blend: if positive, use +0.5; if negative, use -0.5
    andpd   xmm5, xmm4              ; xmm5 = 0.5 if positive, else 0
    andnpd  xmm4, xmm7              ; xmm4 = -0.5 if negative, else 0
    orpd    xmm5, xmm4              ; xmm5 = +-0.5 based on sign
    addsd   xmm0, xmm5              ; xmm0 = x/ln2 +- 0.5
    cvttsd2si rax, xmm0             ; rax = n = trunc(x/ln2 +- 0.5)

    ; Step 3: r = x - n * ln2
    cvtsi2sd xmm0, rax              ; xmm0 = (double)n
    mulsd   xmm0, [rel ln2]         ; xmm0 = n * ln2
    subsd   xmm1, xmm0             ; xmm1 = r = x - n * ln2

    ; Step 6 (early check): Handle underflow/overflow
    mov     rcx, rax
    add     rcx, 1023               ; rcx = n + 1023 (biased exponent)
    cmp     rcx, 0
    jle     .underflow
    cmp     rcx, 2046
    jge     .overflow

    ; Step 4: Evaluate P(r) using Horner's method
    ; P = ((((((c6*r + c5)*r + c4)*r + c3)*r + c2)*r + c1)*r + c0)
    ; xmm1 = r
    movsd   xmm0, [rel c6]          ; xmm0 = c6
    mulsd   xmm0, xmm1              ; c6*r
    addsd   xmm0, [rel c5]          ; c6*r + c5
    mulsd   xmm0, xmm1              ; (c6*r + c5)*r
    addsd   xmm0, [rel c4]          ; ... + c4
    mulsd   xmm0, xmm1
    addsd   xmm0, [rel c3]          ; ... + c3
    mulsd   xmm0, xmm1
    addsd   xmm0, [rel c2]          ; ... + c2
    mulsd   xmm0, xmm1
    addsd   xmm0, [rel one]         ; ... + c1 (1.0)
    mulsd   xmm0, xmm1
    addsd   xmm0, [rel one]         ; ... + c0 (1.0)

    ; Step 5: Reconstruct e^x = 2^n * P(r)
    ; Create 2^n: biased exponent in bits [62:52] of IEEE 754 double
    ; rcx already = n + 1023
    shl     rcx, 52
    movq    xmm1, rcx               ; xmm1 = 2^n as double
    mulsd   xmm0, xmm1              ; xmm0 = P(r) * 2^n = e^x
    ret

.underflow:
    xorpd   xmm0, xmm0              ; return 0.0
    ret

.overflow:
    ; Saturate to largest finite double
    mov     rcx, 2046
    shl     rcx, 52
    mov     rdx, 0x000FFFFFFFFFFFFF  ; mantissa all 1s
    or      rcx, rdx
    movq    xmm0, rcx               ; xmm0 = largest finite double
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
