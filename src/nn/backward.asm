; SPDX-License-Identifier: Apache-2.0
; Copyright (c) 2026-present Steven Baumann
; Original repository: https://github.com/SBNovaScript/nasmlearnsimple
; See LICENSE and NOTICE in the repository root for details.
;
; src/nn/backward.asm
; Backpropagation gradient computation
;
; backward_output(rdi: *output, rsi: *expected, rdx: *hidden_out,
;                 rcx: *grad_w_out, r8: *grad_b_out)
;
; backward_hidden(rdi: *hidden_out, rsi: *input, rdx: *delta_out,
;                 rcx: *w_output, r8: *grad_w_hid, r9: *grad_b_hid)

bits 64
default rel

extern sigmoid_deriv

section .text
global backward_output
global backward_hidden

; ============================================================
; backward_output
; Computes output layer delta and gradients (1 output neuron)
;   MSE gradient omits factor of 2 (absorbed into learning rate)
;   delta = (output - expected) * sigmoid_deriv(output)
;   grad_w_out[i] = delta * hidden_out[i]
;   grad_b_out = delta
; ============================================================
backward_output:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    sub  rsp, 24               ; 8+32+24=64, 64%16==0. Aligned.

    mov  r12, rdx              ; hidden_out
    mov  r13, rcx              ; grad_w_out
    mov  rbx, r8               ; grad_b_out

    ; error = output - expected
    movsd xmm1, [rdi]
    movsd xmm2, xmm1
    subsd xmm1, [rsi]
    movsd [rbp-40], xmm1      ; save error

    movsd xmm0, xmm2
    call  sigmoid_deriv

    ; delta = error * sigmoid'(output)
    mulsd xmm0, [rbp-40]

    movsd [rbx], xmm0         ; grad_b_out = delta

    ; grad_w_out[i] = delta * hidden_out[i]
    movsd xmm1, xmm0
    movsd xmm2, [r12]
    mulsd xmm2, xmm1
    movsd [r13], xmm2

    movsd xmm2, [r12 + 8]
    mulsd xmm2, xmm1
    movsd [r13 + 8], xmm2

    add  rsp, 24
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

; ============================================================
; backward_hidden
; Computes hidden layer gradients (2 hidden neurons)
;   For each hidden neuron j:
;     delta = delta_out * w_output[j] * sigmoid_deriv(hidden_out[j])
;     grad_b_hid[j] = delta
;     grad_w_hid[j*2+i] = delta * input[i]
; ============================================================
backward_hidden:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub  rsp, 24               ; 8+48+24=80, 80%16==0. Aligned.

    mov  rbx, rdi              ; hidden_out
    mov  r12, rsi              ; input
    mov  r13, rcx              ; w_output
    mov  r14, r8               ; grad_w_hid
    mov  r15, r9               ; grad_b_hid

    movsd xmm4, [rdx]         ; delta_out
    movsd [rbp-48], xmm4

    ; --- Hidden neuron 0 ---
    movsd xmm0, xmm4
    mulsd xmm0, [r13]         ; delta_out * w_output[0]
    movsd [rbp-56], xmm0      ; save propagated error

    movsd xmm0, [rbx]         ; hidden_out[0]
    call  sigmoid_deriv

    mulsd xmm0, [rbp-56]
    movsd [r15], xmm0         ; grad_b_hid[0]

    movsd xmm1, xmm0
    mulsd xmm1, [r12]
    movsd [r14], xmm1         ; grad_w_hid[0]

    movsd xmm1, xmm0
    mulsd xmm1, [r12 + 8]
    movsd [r14 + 8], xmm1     ; grad_w_hid[1]

    ; --- Hidden neuron 1 ---
    movsd xmm0, [rbp-48]      ; reload delta_out
    mulsd xmm0, [r13 + 8]     ; delta_out * w_output[1]
    movsd [rbp-56], xmm0      ; save propagated error

    movsd xmm0, [rbx + 8]     ; hidden_out[1]
    call  sigmoid_deriv

    mulsd xmm0, [rbp-56]
    movsd [r15 + 8], xmm0     ; grad_b_hid[1]

    movsd xmm1, xmm0
    mulsd xmm1, [r12]
    movsd [r14 + 16], xmm1    ; grad_w_hid[2]

    movsd xmm1, xmm0
    mulsd xmm1, [r12 + 8]
    movsd [r14 + 24], xmm1    ; grad_w_hid[3]

    add  rsp, 24
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
