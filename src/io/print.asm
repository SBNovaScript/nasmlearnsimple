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
global print_uint64

; print_string(rdi: *str, rsi: len)
; Writes a string to stdout via sys_write.
print_string:
    mov  rdx, rsi
    mov  rsi, rdi
    mov  eax, SYS_write
    mov  edi, STDOUT
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
    ; Entry RSP%16==8, after push: RSP%16==0. Aligned.

    lea  rdi, [rel print_buf]
    call double_to_str

    mov  rdx, rax
    lea  rsi, [rel print_buf]
    mov  eax, SYS_write
    mov  edi, STDOUT
    syscall

    pop  rbp
    ret

; print_uint64(rdi: unsigned integer)
; Converts to decimal ASCII, prints via print_string.
print_uint64:
    push    rbp
    mov     rbp, rsp
    sub     rsp, 32                 ; local buffer (8+40=48, 48%16==0)

    mov     rax, rdi
    lea     r8, [rbp - 1]           ; write pointer (end of buffer)
    xor     ecx, ecx

    test    rax, rax
    jnz     .pu_nonzero
    mov     byte [r8], '0'
    dec     r8
    inc     ecx
    jmp     .pu_print

.pu_nonzero:
.pu_loop:
    test    rax, rax
    jz      .pu_print
    xor     edx, edx
    mov     r9, 10
    div     r9
    add     dl, '0'
    mov     [r8], dl
    dec     r8
    inc     ecx
    jmp     .pu_loop

.pu_print:
    lea     rdi, [r8 + 1]
    mov     esi, ecx
    call    print_string

    add     rsp, 32
    pop     rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
