; ht_block_encoding_avx512.asm
;
; NASM scaffolding for the AVX-512 cleanup encoder hot path.
;
; PR A1 of the cleanup_encode rewrite series: this file exists only to wire
; the NASM toolchain into the build so PR A2 can drop in
; `openhtj2k_avx512_emit_flat` without also having to land the CMake plumbing.
; The only export here is a probe symbol the C++ side can optionally call
; from a debug build to confirm the NASM object linked correctly.
;
; ABI: SysV AMD64 on Linux/macOS, Microsoft x64 on Windows.  Both calling
; conventions return ints in RAX/EAX, so the probe symbol below is portable
; as-is.  Symbol-export syntax differs between object formats:
;   * ELF: `global sym:function` (the :function suffix is ELF-only)
;   * Win64 / Mach-O: `global sym`  (no decorations)
; Mach-O additionally prepends an underscore to C identifiers — handle that
; via the conditional rename below so the same C declaration links on all
; three platforms.

default rel

%ifidn __OUTPUT_FORMAT__, elf64
  %define EXPORT(sym) global sym %+ :function
%elifidn __OUTPUT_FORMAT__, macho64
  %define EXPORT(sym) global _ %+ sym
%else
  %define EXPORT(sym) global sym
%endif

%ifidn __OUTPUT_FORMAT__, macho64
  %define DEFINE_FUNC(sym) _ %+ sym
%else
  %define DEFINE_FUNC(sym) sym
%endif

section .text

; uint32_t openhtj2k_nasm_probe(void);
;
; Returns the PR scaffolding marker 0xA1.  Never called on the hot path —
; reserved for build-system smoke tests and PR A2's bring-up.
EXPORT(openhtj2k_nasm_probe)
DEFINE_FUNC(openhtj2k_nasm_probe):
        mov     eax, 0xA1
        ret

; Non-executable stack note (Linux): silences the GNU linker's
; "missing .note.GNU-stack section implies executable stack" warning.
; The section is only emitted for elf64 output; Win64/Mach-O ignore it.
%ifidn __OUTPUT_FORMAT__, elf64
section .note.GNU-stack noalloc noexec nowrite progbits
%endif
