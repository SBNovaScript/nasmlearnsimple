; src/timing/clock.asm
; Wall-clock timing module using clock_gettime(CLOCK_MONOTONIC).
; Collects min/max/sum/count stats across multiple trials
; and prints formatted results in milliseconds.
;
; Target: Linux x86-64 ELF64, SysV AMD64 ABI
; All exported functions are non-leaf (call print_* or syscall wrappers).

bits 64
default rel

%include "constants.inc"

; Imports from io module
extern print_string
extern print_double
extern print_newline
extern print_uint64

; ============================================================
; Exports
; ============================================================
global timer_start
global timer_stop
global timer_record_trial
global timer_reset_stats
global timer_print_stats

; ============================================================
; Read-only data
; ============================================================
section .rodata
    align 8
    ns_per_sec:     dq 1000000000       ; 10^9
    ns_to_ms_div:   dq 1000000.0        ; 10^6 as double (ns -> ms)
    uint64_max:     dq 0xFFFFFFFFFFFFFFFF

    msg_timing:     db 10, "--- Timing Results ---", 10
    msg_timing_len  equ $ - msg_timing
    msg_trials:     db "Trials:  "
    msg_trials_len  equ $ - msg_trials
    msg_avg:        db "Average: "
    msg_avg_len     equ $ - msg_avg
    msg_min:        db "Min:     "
    msg_min_len     equ $ - msg_min
    msg_max:        db "Max:     "
    msg_max_len     equ $ - msg_max
    msg_ms:         db " ms"
    msg_ms_len      equ $ - msg_ms

; ============================================================
; Uninitialized data
; ============================================================
section .bss
    ; timespec structs: { tv_sec (8 bytes), tv_nsec (8 bytes) }
    ts_start:       resq 2
    ts_stop:        resq 2

    ; Last elapsed time in nanoseconds
    elapsed_ns:     resq 1

    ; Statistics accumulators
    stat_count:     resq 1
    stat_sum_ns:    resq 1
    stat_min_ns:    resq 1
    stat_max_ns:    resq 1

; ============================================================
; Code
; ============================================================
section .text

; ------------------------------------------------------------
; timer_start()
; Captures start timestamp via clock_gettime(CLOCK_MONOTONIC).
; Leaf-ish (syscall, no call instructions).
; ------------------------------------------------------------
timer_start:
    mov     eax, SYS_clock_gettime
    mov     edi, CLOCK_MONOTONIC
    lea     rsi, [rel ts_start]
    syscall
    ret

; ------------------------------------------------------------
; timer_stop()
; Captures stop timestamp, computes elapsed nanoseconds.
; Result stored in [elapsed_ns].
; Leaf-ish (syscall, no call instructions).
; ------------------------------------------------------------
timer_stop:
    mov     eax, SYS_clock_gettime
    mov     edi, CLOCK_MONOTONIC
    lea     rsi, [rel ts_stop]
    syscall

    ; elapsed_ns = (stop.sec - start.sec) * 1e9 + (stop.nsec - start.nsec)
    mov     rax, [rel ts_stop]
    sub     rax, [rel ts_start]
    imul    rax, [rel ns_per_sec]
    mov     rcx, [rel ts_stop + 8]
    sub     rcx, [rel ts_start + 8]
    add     rax, rcx
    mov     [rel elapsed_ns], rax
    ret

; ------------------------------------------------------------
; timer_record_trial()
; Accumulates elapsed_ns into stats (sum, count, min, max).
; Leaf function.
; ------------------------------------------------------------
timer_record_trial:
    mov     rax, [rel elapsed_ns]
    add     [rel stat_sum_ns], rax
    inc     qword [rel stat_count]

    cmp     rax, [rel stat_min_ns]
    jae     .not_new_min
    mov     [rel stat_min_ns], rax
.not_new_min:
    cmp     rax, [rel stat_max_ns]
    jbe     .not_new_max
    mov     [rel stat_max_ns], rax
.not_new_max:
    ret

; ------------------------------------------------------------
; timer_reset_stats()
; Zeros sum/count/max, sets min to UINT64_MAX.
; Leaf function.
; ------------------------------------------------------------
timer_reset_stats:
    xor     eax, eax
    mov     [rel stat_count], rax
    mov     [rel stat_sum_ns], rax
    mov     [rel stat_max_ns], rax
    mov     rax, [rel uint64_max]
    mov     [rel stat_min_ns], rax
    ret

; ------------------------------------------------------------
; timer_print_stats()
; Prints formatted timing results: Trials, Average, Min, Max.
; Calls: print_string, print_double, print_newline, print_uint64, _ns_to_ms
; ------------------------------------------------------------
timer_print_stats:
    push    rbp
    mov     rbp, rsp
    ; Entry RSP%16==8, after push: RSP%16==0. Aligned.

    lea     rdi, [rel msg_timing]
    mov     esi, msg_timing_len
    call    print_string

    ; --- Trials: N ---
    lea     rdi, [rel msg_trials]
    mov     esi, msg_trials_len
    call    print_string

    mov     rdi, [rel stat_count]
    call    print_uint64

    call    print_newline

    ; --- Average: X.XXXX ms ---
    lea     rdi, [rel msg_avg]
    mov     esi, msg_avg_len
    call    print_string

    ; avg_ms = sum_ms / count
    mov     rax, [rel stat_sum_ns]
    call    _ns_to_ms
    movsd   xmm1, xmm0
    mov     rax, [rel stat_count]
    cvtsi2sd xmm0, rax
    divsd   xmm1, xmm0
    movsd   xmm0, xmm1
    call    print_double

    lea     rdi, [rel msg_ms]
    mov     esi, msg_ms_len
    call    print_string
    call    print_newline

    ; --- Min: X.XXXX ms ---
    lea     rdi, [rel msg_min]
    mov     esi, msg_min_len
    call    print_string

    mov     rax, [rel stat_min_ns]
    call    _ns_to_ms
    call    print_double

    lea     rdi, [rel msg_ms]
    mov     esi, msg_ms_len
    call    print_string
    call    print_newline

    ; --- Max: X.XXXX ms ---
    lea     rdi, [rel msg_max]
    mov     esi, msg_max_len
    call    print_string

    mov     rax, [rel stat_max_ns]
    call    _ns_to_ms
    call    print_double

    lea     rdi, [rel msg_ms]
    mov     esi, msg_ms_len
    call    print_string
    call    print_newline

    pop     rbp
    ret

; ============================================================
; Internal helpers (not exported)
; ============================================================

; ------------------------------------------------------------
; _ns_to_ms(rax: nanoseconds) -> xmm0: milliseconds as double
; Leaf function.
; ------------------------------------------------------------
_ns_to_ms:
    cvtsi2sd xmm0, rax
    divsd   xmm0, [rel ns_to_ms_div]
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
