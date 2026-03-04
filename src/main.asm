; src/main.asm
; NASM Learn Simple — Main entry point
; Trains a 2-2-1 neural network to learn XOR via backpropagation.
;
; Register usage in training loop:
;   r12 = epoch counter    (callee-saved, survives all calls)
;   r13 = sample index     (callee-saved, survives all calls)
;   r14 = current input ptr (callee-saved, survives all calls)
;
; Stack alignment: _start enters with RSP%16==0 (ELF ABI).
; All function calls maintain RSP%16==0 before `call`.

bits 64
default rel

%include "constants.inc"

; External module functions
extern print_double
extern print_string
extern print_newline
extern forward_layer
extern backward_output
extern backward_hidden
extern update_weights
extern xor_inputs
extern xor_targets

; ============================================================
; Read-only data
; ============================================================
section .rodata
    align 8
    learning_rate:  dq 2.0
    half_f:         dq 0.5
    one_f:          dq 1.0
    two_f:          dq 2.0
    four_f:         dq 4.0

    msg_title:      db "=== NASM Learn Simple ===", 10
    msg_title_len   equ $ - msg_title
    msg_train:      db "Training XOR network (2-2-1)...", 10
    msg_train_len   equ $ - msg_train
    msg_epoch:      db "Epoch "
    msg_epoch_len   equ $ - msg_epoch
    msg_loss:       db " | Loss: "
    msg_loss_len    equ $ - msg_loss
    msg_results:    db 10, "--- Final Predictions ---", 10
    msg_results_len equ $ - msg_results
    msg_input:      db "Input: ("
    msg_input_len   equ $ - msg_input
    msg_comma:      db ", "
    msg_comma_len   equ $ - msg_comma
    msg_arrow:      db ") -> "
    msg_arrow_len   equ $ - msg_arrow
    msg_expect:     db " (expected: "
    msg_expect_len  equ $ - msg_expect
    msg_paren:      db ")"
    msg_paren_len   equ $ - msg_paren

; ============================================================
; Uninitialized data
; ============================================================
section .bss
    ; Network parameters
    w_hidden:       resq 4      ; 2x2 weight matrix (row-major)
    b_hidden:       resq 2      ; 2 biases
    w_output:       resq 2      ; 1x2 weight matrix
    b_output:       resq 1      ; 1 bias

    ; Activation scratch
    hidden_out:     resq 2      ; hidden layer outputs
    net_output:     resq 1      ; network output

    ; Gradient scratch
    grad_w_out:     resq 2
    grad_b_out:     resq 1
    grad_w_hid:     resq 4
    grad_b_hid:     resq 2
    delta_hidden:   resq 2

    ; PRNG state
    lcg_state:      resq 1

    ; Loss accumulator
    epoch_loss:     resq 1

; ============================================================
; Code
; ============================================================
section .text
global _start

; ------------------------------------------------------------
; init_rng: Seed PRNG from TSC. Leaf function.
; ------------------------------------------------------------
init_rng:
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    mov     [rel lcg_state], rax
    ret

; ------------------------------------------------------------
; next_random() -> xmm0: random double in [-0.5, 0.5)
; LCG with Knuth constants. Leaf function.
; Range [-0.5, 0.5) approximates Xavier init for fan_in=2,
; reducing sigmoid saturation and improving convergence.
; ------------------------------------------------------------
next_random:
    mov     rax, [rel lcg_state]
    mov     rcx, 6364136223846793005
    imul    rax, rcx
    mov     rcx, 1442695040888963407
    add     rax, rcx
    mov     [rel lcg_state], rax

    ; Convert upper bits to double in [0, 1)
    shr     rax, 11                 ; 53-bit unsigned value
    cvtsi2sd xmm0, rax
    mov     rax, 1
    shl     rax, 53
    cvtsi2sd xmm1, rax
    divsd   xmm0, xmm1             ; [0, 1)

    ; Scale to [-0.5, 0.5)
    subsd   xmm0, [rel half_f]     ; [-0.5, 0.5)
    ret

; ------------------------------------------------------------
; init_weights: Fill all weights/biases with random values.
; Calls: init_rng, next_random
; ------------------------------------------------------------
init_weights:
    push    rbp
    mov     rbp, rsp
    push    r12
    push    r13
    ; 3 pushes: RSP moved 24 from entry. Entry RSP%16==8.
    ; 8+24=32, 32%16==0. Aligned for calls.

    call    init_rng

    ; w_hidden: 4 doubles
    lea     r12, [rel w_hidden]
    mov     r13, 4
.iw_wh:
    call    next_random
    movsd   [r12], xmm0
    add     r12, 8
    dec     r13
    jnz     .iw_wh

    ; b_hidden: 2 doubles
    lea     r12, [rel b_hidden]
    mov     r13, 2
.iw_bh:
    call    next_random
    movsd   [r12], xmm0
    add     r12, 8
    dec     r13
    jnz     .iw_bh

    ; w_output: 2 doubles
    lea     r12, [rel w_output]
    mov     r13, 2
.iw_wo:
    call    next_random
    movsd   [r12], xmm0
    add     r12, 8
    dec     r13
    jnz     .iw_wo

    ; b_output: 1 double
    call    next_random
    movsd   [rel b_output], xmm0

    pop     r13
    pop     r12
    pop     rbp
    ret

; ------------------------------------------------------------
; print_int(rdi: unsigned integer)
; Converts to decimal ASCII, prints via print_string.
; Calls: print_string
; ------------------------------------------------------------
print_int:
    push    rbp
    mov     rbp, rsp
    sub     rsp, 32                 ; local buffer (32 bytes)
    ; 1 push + 32 sub = 40 from entry. Entry RSP%16==8.
    ; 8+40=48, 48%16==0. Aligned for calls.

    mov     rax, rdi
    lea     r8, [rbp - 1]           ; write pointer (end of buffer)
    xor     ecx, ecx                ; digit count

    test    rax, rax
    jnz     .pi_nonzero
    mov     byte [r8], '0'
    dec     r8
    inc     ecx
    jmp     .pi_print

.pi_nonzero:
.pi_loop:
    test    rax, rax
    jz      .pi_print
    xor     edx, edx
    mov     r9, 10
    div     r9
    add     dl, '0'
    mov     [r8], dl
    dec     r8
    inc     ecx
    jmp     .pi_loop

.pi_print:
    lea     rdi, [r8 + 1]           ; start of digit string
    mov     esi, ecx                ; length
    call    print_string

    add     rsp, 32
    pop     rbp
    ret

; ============================================================
; _start: Main entry point
; ============================================================
_start:
    ; RSP%16==0 at ELF entry point

    ; --- Print title ---
    lea     rdi, [rel msg_title]
    mov     esi, msg_title_len
    call    print_string

    lea     rdi, [rel msg_train]
    mov     esi, msg_train_len
    call    print_string

    ; --- Initialize weights ---
    call    init_weights

    ; --- Training loop ---
    xor     r12d, r12d              ; r12 = epoch = 0

.epoch_loop:
    cmp     r12, MAX_EPOCHS
    jge     .training_done

    ; Zero epoch loss
    xorpd   xmm0, xmm0
    movsd   [rel epoch_loss], xmm0

    ; Iterate 4 XOR samples
    xor     r13d, r13d              ; r13 = sample index

.sample_loop:
    cmp     r13, 4
    jge     .sample_done

    ; Compute input pointer: xor_inputs + sample * 16
    mov     rax, r13
    shl     rax, 4
    lea     r14, [rel xor_inputs]
    add     r14, rax                ; r14 = &input[sample] (preserved across calls)

    ; --- Forward pass: hidden layer ---
    ; forward_layer(input, w_hidden, b_hidden, hidden_out, 2, 2)
    mov     rdi, r14
    lea     rsi, [rel w_hidden]
    lea     rdx, [rel b_hidden]
    lea     rcx, [rel hidden_out]
    mov     r8, INPUT_SIZE
    mov     r9, HIDDEN_SIZE
    call    forward_layer

    ; --- Forward pass: output layer ---
    ; forward_layer(hidden_out, w_output, b_output, net_output, 2, 1)
    lea     rdi, [rel hidden_out]
    lea     rsi, [rel w_output]
    lea     rdx, [rel b_output]
    lea     rcx, [rel net_output]
    mov     r8, HIDDEN_SIZE
    mov     r9, OUTPUT_SIZE
    call    forward_layer

    ; --- Accumulate loss: (output - expected)^2 ---
    movsd   xmm0, [rel net_output]
    mov     rax, r13
    lea     rcx, [rel xor_targets]
    movsd   xmm1, [rcx + rax*8]
    subsd   xmm0, xmm1
    mulsd   xmm0, xmm0
    addsd   xmm0, [rel epoch_loss]
    movsd   [rel epoch_loss], xmm0

    ; --- Backward pass: output layer ---
    ; backward_output(output, expected, hidden_out, grad_w_out, grad_b_out)
    lea     rdi, [rel net_output]
    mov     rax, r13
    lea     rsi, [rel xor_targets]
    lea     rsi, [rsi + rax*8]
    lea     rdx, [rel hidden_out]
    lea     rcx, [rel grad_w_out]
    lea     r8, [rel grad_b_out]
    call    backward_output

    ; --- Backward pass: hidden layer ---
    ; backward_hidden(hidden_out, input, delta_out, w_output,
    ;                 grad_w_hid, grad_b_hid, delta_hidden)
    ; 7th arg (delta_hidden) goes on stack
    lea     rdi, [rel hidden_out]
    mov     rsi, r14                ; input pointer (callee-saved)
    lea     rdx, [rel grad_b_out]   ; delta_out value stored here
    lea     rcx, [rel w_output]
    lea     r8, [rel grad_w_hid]
    lea     r9, [rel grad_b_hid]
    lea     rax, [rel delta_hidden]
    sub     rsp, 8                  ; alignment padding
    push    rax                     ; 7th arg at [rsp], RSP%16==0
    call    backward_hidden
    add     rsp, 16                 ; clean up 7th arg + padding

    ; --- Update weights: output layer ---
    lea     rdi, [rel w_output]
    lea     rsi, [rel grad_w_out]
    mov     edx, HIDDEN_SIZE
    movsd   xmm0, [rel learning_rate]
    call    update_weights

    ; --- Update biases: output layer ---
    lea     rdi, [rel b_output]
    lea     rsi, [rel grad_b_out]
    mov     edx, OUTPUT_SIZE
    movsd   xmm0, [rel learning_rate]
    call    update_weights

    ; --- Update weights: hidden layer ---
    lea     rdi, [rel w_hidden]
    lea     rsi, [rel grad_w_hid]
    mov     edx, 4                  ; HIDDEN_SIZE * INPUT_SIZE
    movsd   xmm0, [rel learning_rate]
    call    update_weights

    ; --- Update biases: hidden layer ---
    lea     rdi, [rel b_hidden]
    lea     rsi, [rel grad_b_hid]
    mov     edx, HIDDEN_SIZE
    movsd   xmm0, [rel learning_rate]
    call    update_weights

    inc     r13
    jmp     .sample_loop

.sample_done:
    ; Average loss over 4 samples
    movsd   xmm0, [rel epoch_loss]
    divsd   xmm0, [rel four_f]
    movsd   [rel epoch_loss], xmm0

    ; Print every PRINT_EVERY epochs
    mov     rax, r12
    xor     edx, edx
    mov     rcx, PRINT_EVERY
    div     rcx
    test    rdx, rdx
    jnz     .skip_print

    ; Print "Epoch N | Loss: X.XXXX"
    lea     rdi, [rel msg_epoch]
    mov     esi, msg_epoch_len
    call    print_string

    mov     rdi, r12
    call    print_int

    lea     rdi, [rel msg_loss]
    mov     esi, msg_loss_len
    call    print_string

    movsd   xmm0, [rel epoch_loss]
    call    print_double

    call    print_newline

.skip_print:
    inc     r12
    jmp     .epoch_loop

; ============================================================
; Training complete — print final predictions
; ============================================================
.training_done:
    lea     rdi, [rel msg_results]
    mov     esi, msg_results_len
    call    print_string

    xor     r13d, r13d              ; sample index

.result_loop:
    cmp     r13, 4
    jge     .exit

    ; Compute input pointer
    mov     rax, r13
    shl     rax, 4
    lea     r14, [rel xor_inputs]
    add     r14, rax

    ; Forward pass: hidden layer
    mov     rdi, r14
    lea     rsi, [rel w_hidden]
    lea     rdx, [rel b_hidden]
    lea     rcx, [rel hidden_out]
    mov     r8, INPUT_SIZE
    mov     r9, HIDDEN_SIZE
    call    forward_layer

    ; Forward pass: output layer
    lea     rdi, [rel hidden_out]
    lea     rsi, [rel w_output]
    lea     rdx, [rel b_output]
    lea     rcx, [rel net_output]
    mov     r8, HIDDEN_SIZE
    mov     r9, OUTPUT_SIZE
    call    forward_layer

    ; Print: "Input: (x1, x2) -> output (expected: target)"
    lea     rdi, [rel msg_input]
    mov     esi, msg_input_len
    call    print_string

    movsd   xmm0, [r14]            ; input[0]
    call    print_double

    lea     rdi, [rel msg_comma]
    mov     esi, msg_comma_len
    call    print_string

    movsd   xmm0, [r14 + 8]        ; input[1]
    call    print_double

    lea     rdi, [rel msg_arrow]
    mov     esi, msg_arrow_len
    call    print_string

    movsd   xmm0, [rel net_output]
    call    print_double

    lea     rdi, [rel msg_expect]
    mov     esi, msg_expect_len
    call    print_string

    mov     rax, r13
    lea     rcx, [rel xor_targets]
    movsd   xmm0, [rcx + rax*8]
    call    print_double

    lea     rdi, [rel msg_paren]
    mov     esi, msg_paren_len
    call    print_string

    call    print_newline

    inc     r13
    jmp     .result_loop

.exit:
    mov     eax, SYS_exit
    xor     edi, edi
    syscall

section .note.GNU-stack noalloc noexec nowrite progbits
