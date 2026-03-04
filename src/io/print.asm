; src/io/print.asm
; Print routines using Linux syscalls

bits 64
default rel

%include "constants.inc"

extern double_to_str

section .data
    newline_char: db 10

section .bss
    print_buf: resb 64

section .text
global print_string
global print_newline
global print_double

; print_string(rdi: *str, rsi: len)
; Writes a string to stdout via sys_write.
print_string:
    ; No callee-saved registers needed, no calls made
    mov  rdx, rsi           ; rdx = length (arg3 for sys_write)
    mov  rsi, rdi           ; rsi = buffer pointer (arg2 for sys_write)
    mov  eax, SYS_write     ; syscall number
    mov  edi, STDOUT        ; fd = stdout
    syscall
    ret

; print_newline()
; Writes a newline character to stdout.
print_newline:
    mov  eax, SYS_write
    mov  edi, STDOUT
    lea  rsi, [rel newline_char]
    mov  edx, 1
    syscall
    ret

; print_double(xmm0: double)
; Converts double to string and writes to stdout.
print_double:
    push rbp
    mov  rbp, rsp
    ; At entry RSP % 16 == 8, after push rbp -> RSP % 16 == 0
    ; Before calling double_to_str, RSP must be 16-byte aligned.
    ; RSP is currently 16-byte aligned, and call will push 8, making it
    ; RSP % 16 == 8 inside callee — that's correct for SysV ABI.

    ; Prepare call to double_to_str(xmm0, rdi)
    ; xmm0 already has the double value
    lea  rdi, [rel print_buf]
    call double_to_str

    ; rax = length of string written to print_buf
    mov  rdx, rax           ; rdx = length
    lea  rsi, [rel print_buf]
    mov  eax, SYS_write
    mov  edi, STDOUT
    syscall

    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
