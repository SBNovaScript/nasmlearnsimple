; SPDX-License-Identifier: Apache-2.0
; Copyright (c) 2026-present Steven Baumann
; Original repository: https://github.com/SBNovaScript/nasmlearnsimple
; See LICENSE and NOTICE in the repository root for details.
;
; src/io/string.asm
; Double-to-ASCII conversion with 4 decimal places

bits 64
default rel

section .data
    align 8
    const_10000: dq 10000.0
    const_0:     dq 0.0

section .text
global double_to_str

; double_to_str(xmm0: double, rdi: *buf) -> rax: length
; Converts a double to decimal ASCII string with 4 decimal places.
; Follows SysV AMD64 ABI.
double_to_str:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    ; 4 pushes total (rbp + 3 callee-saved) -> RSP % 16 == 8 at entry,
    ; after 4 pushes (32 bytes) -> RSP % 16 == 8 + 32 = 40 -> 40 % 16 == 8
    ; Need one more push or sub to align, but we don't call anything, so it's fine.
    sub  rsp, 32            ; local scratch space (temp buffer for integer digits)

    mov  r12, rdi           ; r12 = output buffer pointer
    xor  r13d, r13d         ; r13 = write offset into buffer

    ; ---- Handle sign ----
    movsd xmm1, [rel const_0]
    ucomisd xmm0, xmm1
    jae  .not_negative
    ; Write '-'
    mov  byte [r12 + r13], '-'
    inc  r13
    ; Negate: xmm0 = 0.0 - xmm0
    movsd xmm1, xmm0
    xorpd xmm0, xmm0
    subsd xmm0, xmm1
.not_negative:

    ; ---- Extract integer part ----
    ; xmm0 = |value|
    cvttsd2si rax, xmm0    ; rax = integer part (truncated toward zero)
    mov  r14, rax           ; r14 = integer part

    ; Subtract integer part to get fractional part
    cvtsi2sd xmm1, rax     ; xmm1 = (double)integer_part
    subsd xmm0, xmm1       ; xmm0 = fractional part

    ; ---- Convert integer part to ASCII digits ----
    ; Write digits in reverse into temp buffer on stack, then copy forward.
    ; temp buffer is at [rsp] .. [rsp+19]
    lea  rbx, [rsp]         ; rbx = temp buffer base
    xor  ecx, ecx           ; ecx = digit count

    mov  rax, r14           ; rax = integer part
    test rax, rax
    jnz  .int_digit_loop
    ; Special case: integer part is 0
    mov  byte [rbx], '0'
    inc  ecx
    jmp  .int_digits_done

.int_digit_loop:
    xor  edx, edx
    mov  r8, 10
    div  r8                 ; rax = quotient, rdx = remainder
    add  dl, '0'
    mov  byte [rbx + rcx], dl
    inc  ecx
    test rax, rax
    jnz  .int_digit_loop

.int_digits_done:
    ; Copy digits in reverse order (they're stored least-significant first)
    ; ecx = number of digits
    mov  eax, ecx
    dec  eax                ; eax = index of last digit in temp
.copy_int_digits:
    mov  dl, byte [rbx + rax]
    mov  byte [r12 + r13], dl
    inc  r13
    dec  eax
    jns  .copy_int_digits

    ; ---- Write decimal point ----
    mov  byte [r12 + r13], '.'
    inc  r13

    ; ---- Fractional part: 4 decimal places with leading zeros ----
    ; xmm0 = fractional part (>= 0, < 1)
    mulsd xmm0, [rel const_10000]  ; xmm0 = frac * 10000
    ; Round to nearest integer to avoid floating-point truncation issues
    roundsd xmm0, xmm0, 0          ; round to nearest even
    cvttsd2si rax, xmm0            ; rax = 4-digit integer (0..9999)

    ; Digit 1: rax / 1000
    xor  edx, edx
    mov  r8, 1000
    div  r8
    add  al, '0'
    mov  byte [r12 + r13], al
    inc  r13

    ; Digit 2: rdx / 100
    mov  rax, rdx
    xor  edx, edx
    mov  r8, 100
    div  r8
    add  al, '0'
    mov  byte [r12 + r13], al
    inc  r13

    ; Digit 3: rdx / 10
    mov  rax, rdx
    xor  edx, edx
    mov  r8, 10
    div  r8
    add  al, '0'
    mov  byte [r12 + r13], al
    inc  r13

    ; Digit 4: rdx
    add  dl, '0'
    mov  byte [r12 + r13], dl
    inc  r13

    ; ---- Return length ----
    mov  rax, r13

    add  rsp, 32
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
