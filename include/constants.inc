; include/constants.inc
; Shared constants for NASM Learn Simple

%ifndef CONSTANTS_INC
%define CONSTANTS_INC

; --- Syscall numbers ---
%define SYS_write   1
%define SYS_exit    60
%define STDOUT      1

; --- Network dimensions ---
%define INPUT_SIZE   2
%define HIDDEN_SIZE  2
%define OUTPUT_SIZE  1

; --- Training parameters ---
%define MAX_EPOCHS   50000
%define PRINT_EVERY  10000

%endif
