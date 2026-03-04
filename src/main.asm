; src/main.asm
; Entry point — stub for build verification

bits 64
default rel

%include "constants.inc"

section .text
global _start

_start:
    mov eax, SYS_exit
    xor edi, edi
    syscall

section .note.GNU-stack noalloc noexec nowrite progbits
