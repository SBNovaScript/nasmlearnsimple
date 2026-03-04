; src/main.asm
; Entry point — tests I/O and math modules

bits 64
default rel

%include "constants.inc"

extern print_string
extern print_double
extern print_newline
extern sigmoid

section .data
    align 8
    test_msg:       db "I/O test: "
    test_msg_len    equ $ - test_msg
    test_val:       dq 3.1415

    sig_msg:        db "sigmoid(0) = "
    sig_msg_len     equ $ - sig_msg
    zero_val:       dq 0.0

section .text
global _start

_start:
    ; Print "I/O test: "
    lea  rdi, [rel test_msg]
    mov  rsi, test_msg_len
    call print_string

    ; Print 3.1415
    movsd xmm0, [rel test_val]
    call print_double

    ; Print newline
    call print_newline

    ; Print "sigmoid(0) = "
    lea  rdi, [rel sig_msg]
    mov  rsi, sig_msg_len
    call print_string

    ; Compute sigmoid(0.0)
    movsd xmm0, [rel zero_val]
    call sigmoid

    ; Print result
    call print_double

    ; Print newline
    call print_newline

    ; Exit 0
    mov  eax, SYS_exit
    xor  edi, edi
    syscall

section .note.GNU-stack noalloc noexec nowrite progbits
