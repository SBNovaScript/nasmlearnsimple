; src/nn/backward.asm
; Backpropagation gradient computation
;
; backward_output(rdi: *output, rsi: *expected, rdx: *hidden_out,
;                 rcx: *grad_w_out, r8: *grad_b_out)
;
; backward_hidden(rdi: *hidden_out, rsi: *input, rdx: *delta_out,
;                 rcx: *w_output, r8: *grad_w_hid, r9: *grad_b_hid,
;                 [rbp+16]: *delta_hidden)

bits 64
default rel

extern sigmoid_deriv

section .text
global backward_output
global backward_hidden

; ============================================================
; backward_output
; Computes output layer delta and gradients (1 output neuron)
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
    ; 4 pushes total: RSP moved by 32 from entry. Entry RSP%16==8.
    ; 8+32+24=64, 64%16==0. Aligned. Locals at [rbp-25]..[rbp-48].
    sub  rsp, 24

    mov  r12, rdx              ; hidden_out
    mov  r13, rcx              ; grad_w_out
    mov  rbx, r8               ; grad_b_out

    ; Compute error = output[0] - expected[0]
    movsd xmm1, [rdi]         ; output[0]
    movsd xmm2, xmm1          ; save output for sigmoid_deriv
    subsd xmm1, [rsi]         ; error = output - expected
    movsd [rbp-40], xmm1      ; save error to stack (using aligned local space)

    ; sigmoid_deriv(output[0]) -> s * (1-s)
    movsd xmm0, xmm2          ; pass output value
    call  sigmoid_deriv        ; xmm0 = deriv

    ; delta = error * deriv
    mulsd xmm0, [rbp-40]      ; xmm0 = delta

    ; grad_b_out[0] = delta
    movsd [rbx], xmm0

    ; grad_w_out[i] = delta * hidden_out[i]
    movsd xmm1, xmm0          ; delta
    movsd xmm2, [r12]         ; hidden_out[0]
    mulsd xmm2, xmm1
    movsd [r13], xmm2         ; grad_w_out[0]

    movsd xmm2, [r12 + 8]     ; hidden_out[1]
    mulsd xmm2, xmm1
    movsd [r13 + 8], xmm2     ; grad_w_out[1]

    add  rsp, 24
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

; ============================================================
; backward_hidden
; Computes hidden layer deltas and gradients (2 hidden neurons)
;   For each hidden neuron j:
;     delta_hidden[j] = delta_out * w_output[j] * sigmoid_deriv(hidden_out[j])
;     grad_b_hid[j] = delta_hidden[j]
;     grad_w_hid[j*2+i] = delta_hidden[j] * input[i]
; ============================================================
backward_hidden:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    ; 6 pushes: RSP moved by 48. Entry RSP%16==8. 8+48=56. 56%16==8.
    ; sub rsp, 56 for locals + alignment. 8+48+56=112. 112%16==0. Aligned!
    sub  rsp, 56               ; locals + alignment

    mov  r10, rdi              ; hidden_out
    mov  r11, rsi              ; input
    mov  r12, rdx              ; delta_out ptr
    mov  r13, rcx              ; w_output
    mov  r14, r8               ; grad_w_hid
    mov  r15, r9               ; grad_b_hid
    mov  rbx, [rbp + 16]      ; delta_hidden (7th arg from stack)

    ; Load output delta value
    movsd xmm4, [r12]         ; delta_out[0]
    movsd [rbp-56], xmm4      ; save delta_out to local

    ; --- Hidden neuron 0 ---
    ; propagated = delta_out * w_output[0]
    movsd xmm0, xmm4
    mulsd xmm0, [r13]         ; delta_out * w_output[0]
    movsd [rbp-64], xmm0      ; save propagated error

    ; sigmoid_deriv(hidden_out[0])
    movsd xmm0, [r10]         ; hidden_out[0]
    ; r10, r11 are caller-saved — save before call
    mov  [rbp-72], r10
    mov  [rbp-80], r11
    call  sigmoid_deriv        ; xmm0 = deriv
    mov  r10, [rbp-72]
    mov  r11, [rbp-80]

    ; delta_hidden[0] = propagated * deriv
    mulsd xmm0, [rbp-64]      ; delta_hidden[0]
    movsd [rbx], xmm0         ; store delta_hidden[0]
    movsd [r15], xmm0         ; grad_b_hid[0]

    ; grad_w_hid[0] = delta_hidden[0] * input[0]
    movsd xmm1, xmm0
    mulsd xmm1, [r11]
    movsd [r14], xmm1

    ; grad_w_hid[1] = delta_hidden[0] * input[1]
    movsd xmm1, xmm0
    mulsd xmm1, [r11 + 8]
    movsd [r14 + 8], xmm1

    ; --- Hidden neuron 1 ---
    movsd xmm0, [rbp-56]      ; reload delta_out
    mulsd xmm0, [r13 + 8]     ; delta_out * w_output[1]
    movsd [rbp-64], xmm0      ; save propagated error

    movsd xmm0, [r10 + 8]     ; hidden_out[1]
    mov  [rbp-72], r10
    mov  [rbp-80], r11
    call  sigmoid_deriv
    mov  r10, [rbp-72]
    mov  r11, [rbp-80]

    mulsd xmm0, [rbp-64]      ; delta_hidden[1]
    movsd [rbx + 8], xmm0     ; store delta_hidden[1]
    movsd [r15 + 8], xmm0     ; grad_b_hid[1]

    ; grad_w_hid[2] = delta_hidden[1] * input[0]
    movsd xmm1, xmm0
    mulsd xmm1, [r11]
    movsd [r14 + 16], xmm1

    ; grad_w_hid[3] = delta_hidden[1] * input[1]
    movsd xmm1, xmm0
    mulsd xmm1, [r11 + 8]
    movsd [r14 + 24], xmm1

    add  rsp, 56
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
