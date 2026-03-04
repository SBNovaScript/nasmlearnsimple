; src/main.asm
; Entry point — tests I/O module

bits 64
default rel

%include "constants.inc"

extern print_string
extern print_double
extern print_newline

section .data
    align 8
    test_msg:    db "I/O test: "
    test_msg_len equ $ - test_msg
    test_val:    dq 3.1415

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

    ; Exit 0
    mov  eax, SYS_exit
    xor  edi, edi
    syscall

section .note.GNU-stack noalloc noexec nowrite progbits
