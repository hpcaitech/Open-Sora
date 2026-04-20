# host-cpu-c-abi.m4 serial 15
dnl Copyright (C) 2002-2022 Free Software Foundation, Inc.
dnl This file is free software; the Free Software Foundation
dnl gives unlimited permission to copy and/or distribute it,
dnl with or without modifications, as long as this notice is preserved.

dnl From Bruno Haible and Sam Steingold.

dnl Sets the HOST_CPU variable to the canonical name of the CPU.
dnl Sets the HOST_CPU_C_ABI variable to the canonical name of the CPU with its
dnl C language ABI (application binary interface).
dnl Also defines __${HOST_CPU}__ and __${HOST_CPU_C_ABI}__ as C macros in
dnl config.h.
dnl
dnl This canonical name can be used to select a particular assembly language
dnl source file that will interoperate with C code on the given host.
dnl
dnl For example:
dnl * 'i386' and 'sparc' are different canonical names, because code for i386
dnl   will not run on SPARC CPUs and vice versa. They have different
dnl   instruction sets.
dnl * 'sparc' and 'sparc64' are different canonical names, because code for
dnl   'sparc' and code for 'sparc64' cannot be linked together: 'sparc' code
dnl   contains 32-bit instructions, whereas 'sparc64' code contains 64-bit
dnl   instructions. A process on a SPARC CPU can be in 32-bit mode or in 64-bit
dnl   mode, but not both.
dnl * 'mips' and 'mipsn32' are different canonical names, because they use
dnl   different argument passing and return conventions for C functions, and
dnl   although the instruction set of 'mips' is a large subset of the
dnl   instruction set of 'mipsn32'.
dnl * 'mipsn32' and 'mips64' are different canonical names, because they use
dnl   different sizes for the C types like 'int' and 'void *', and although
dnl   the instruction sets of 'mipsn32' and 'mips64' are the same.
dnl * The same canonical name is used for different endiannesses. You can
dnl   determine the endianness through preprocessor symbols:
dnl   - 'arm': test __ARMEL__.
dnl   - 'mips', 'mipsn32', 'mips64': test _MIPSEB vs. _MIPSEL.
dnl   - 'powerpc64': test _BIG_ENDIAN vs. _LITTLE_ENDIAN.
dnl * The same name 'i386' is used for CPUs of type i386, i486, i586
dnl   (Pentium), AMD K7, Pentium II, Pentium IV, etc., because
dnl   - Instructions that do not exist on all of these CPUs (cmpxchg,
dnl     MMX, SSE, SSE2, 3DNow! etc.) are not frequently used. If your
dnl     assembly language source files use such instructions, you will
dnl     need to make the distinction.
dnl   - Speed of execution of the common instruction set is reasonable across
dnl     the entire family of CPUs. If you have assembly language source files
dnl     that are optimized for particular CPU types (like GNU gmp has), you
dnl     will need to make the distinction.
dnl   See <https://en.wikipedia.org/wiki/X86_instruction_listings>.
AC_DEFUN([gl_HOST_CPU_C_ABI],
[
  AC_REQUIRE([AC_CANONICAL_HOST])
  AC_REQUIRE([gl_C_ASM])
  AC_CACHE_CHECK([host CPU and C ABI], [gl_cv_host_cpu_c_abi],
    [case "$host_cpu" in

changequote(,)dnl
       i[34567]86 )
changequote([,])dnl
         gl_cv_host_cpu_c_abi=i386
         ;;

       x86_64 )
         # On x86_64 systems, the C compiler may be generating code in one of
         # these ABIs:
         # - 64-bit instruction set, 64-bit pointers, 64-bit 'long': x86_64.
         # - 64-bit instruction set, 64-bit pointers, 32-bit 'long': x86_64
         #   with native Windows (mingw, MSVC).
         # - 64-bit instruction set, 32-bit pointers, 32-bit 'long': x86_64-x32.
         # - 32-bit instruction set, 32-bit pointers, 32-bit 'long': i386.
         AC_COMPILE_IFELSE(
           [AC_LANG_SOURCE(
              [[#if (defined __x86_64__ || defined __amd64__ \
                     || defined _M_X64 || defined _M_AMD64)
                 int ok;
                #else
                 error fail
                #endif
              ]])],
           [AC_COMPILE_IFELSE(
              [AC_LANG_SOURCE(
                 [[#if defined __ILP32__ || defined _ILP32
                    int ok;
                   #else
                    error fail
                   #endif
                 ]])],
              [gl_cv_host_cpu_c_abi=x86_64-x32],
              [gl_cv_host_cpu_c_abi=x86_64])],
           [gl_cv_host_cpu_c_abi=i386])
         ;;

changequote(,)dnl
       alphaev[4-8] | alphaev56 | alphapca5[67] | alphaev6[78] )
changequote([,])dnl
         gl_cv_host_cpu_c_abi=alpha
         ;;

       arm* | aarch64 )
         # Assume arm with EABI.
         # On arm64 systems, the C compiler may be generating code in one of
         # these ABIs:
         # - aarch64 instruction set, 64-bit pointers, 64-bit 'long': arm64.
         # - aarch64 instruction set, 32-bit pointers, 32-bit 'long': arm64-ilp32.
         # - 32-bit instruction set, 32-bit pointers, 32-bit 'long': arm or armhf.
         AC_COMPILE_IFELSE(
           [AC_LANG_SOURCE(
              [[#ifdef __aarch64__
                 int ok;
                #else
                 error fail
                #endif
              ]])],
           [AC_COMPILE_IFELSE(
              [AC_LANG_SOURCE(
                [[#if defined __ILP32__ || defined _ILP32
                   int ok;
                  #else
                   error fail
                  #endif
                ]])],
              [gl_cv_host_cpu_c_abi=arm64-ilp32],
              [gl_cv_host_cpu_c_abi=arm64])],
           [# Don't distinguish little-endian and big-endian arm, since they
            # don't require different machine code for simple operations and
            # since the user can distinguish them through the preprocessor
            # defines __ARMEL__ vs. __ARMEB__.
            # But distinguish arm which passes floating-point arguments and
            # return values in integer registers (r0, r1, ...) - this is
            # gcc -mfloat-abi=soft or gcc -mfloat-abi=softfp - from arm which
            # passes them in float registers (s0, s1, ...) and double registers
            # (d0, d1, ...) - this is gcc -mfloat-abi=hard. GCC 4.6 or newer
            # sets the preprocessor defines __ARM_PCS (for the first case) and
            # __ARM_PCS_VFP (for the second case), but older GCC does not.
            echo 'double ddd; void func (double dd) { ddd = dd; }' > conftest.c
            # Look for a reference to the register d0 in the .s file.
            AC_TRY_COMMAND(${CC-cc} $CFLAGS $CPPFLAGS $gl_c_asm_opt conftest.c) >/dev/null 2>&1
            if LC_ALL=C grep 'd0,' conftest.$gl_asmext >/dev/null; then
              gl_cv_host_cpu_c_abi=armhf
            else
              gl_cv_host_cpu_c_abi=arm
            fi
            rm -f conftest*
           ])
         ;;

       hppa1.0 | hppa1.1 | hppa2.0* | hppa64 )
         # On hppa, the C compiler may be generating 32-bit code or 64-bit
         # code. In the latter case, it defines _LP64 and __LP64__.
         AC_COMPILE_IFELSE(
           [AC_LANG_SOURCE(
              [[#ifdef __LP64__
                 int ok;
                #else
                 error fail
                #endif
              ]])],
           [gl_cv_host_cpu_c_abi=hppa64],
           [gl_cv_host_cpu_c_abi=hppa])
         ;;

       ia64* )
         # On ia64 on HP-UX, the C compiler may be generating 64-bit code or
         # 32-bit code. In the latter case, it defines _ILP32.
         AC_COMPILE_IFELSE(
           [AC_LANG_SOURCE(
              [[#ifdef _ILP32
                 int ok;
                #else
                 error fail
                #endif
              ]])],
           [gl_cv_host_cpu_c_abi=ia64-ilp32],
           [gl_cv_host_cpu_c_abi=ia64])
         ;;

       mips* )
         # We should also check for (_MIPS_SZPTR == 64), but gcc keeps this
         # at 32.
         AC_COMPILE_IFELSE(
           [AC_LANG_SOURCE(
              [[#if defined _MIPS_SZLONG && (_MIPS_SZLONG == 64)
                 int ok;
                #else
                 error fail
                #endif
              ]])],
           [gl_cv_host_cpu_c_abi=mips64],
           [# In the n32 ABI, _ABIN32 is defined, _ABIO32 is not defined (but
            # may later get defined by <sgidefs.h>), and _MIPS_SIM == _ABIN32.
            # In the 32 ABI, _ABIO32 is defined, _ABIN32 is not defined (but
            # may later get defined by <sgidefs.h>), and _MIPS_SIM == _ABIO32.
            AC_COMPILE_IFELSE(
              [AC_LANG_SOURCE(
                 [[#if (_MIPS_SIM == _ABIN32)
                    int ok;
                   #else
                    error fail
                   #endif
                 ]])],
              [gl_cv_host_cpu_c_abi=mipsn32],
              [gl_cv_host_cpu_c_abi=mips])])
         ;;

       powerpc* )
         # Different ABIs are in use on AIX vs. Mac OS X vs. Linux,*BSD.
         # No need to distinguish them here; the caller may distinguish
         # them based on the OS.
         # On powerpc64 systems, the C compiler may still be generating
         # 32-bit code. And on powerpc-ibm-aix systems, the C compiler may
         # be generating 64-bit code.
         AC_COMPILE_IFELSE(
           [AC_LANG_SOURCE(
              [[#if defined __powerpc64__ || defined __LP64__
                 int ok;
                #else
                 error fail
                #endif
              ]])],
           [# On powerpc64, there are two ABIs on Linux: The AIX compatible
            # one and the ELFv2 one. The latter defines _CALL_ELF=2.
            AC_COMPILE_IFELSE(
              [AC_LANG_SOURCE(
                 [[#if defined _CALL_ELF && _CALL_ELF == 2
                    int ok;
                   #else
                    error fail
                   #endif
                 ]])],
              [gl_cv_host_cpu_c_abi=powerpc64-elfv2],
              [gl_cv_host_cpu_c_abi=powerpc64])
           ],
           [gl_cv_host_cpu_c_abi=powerpc])
         ;;

       rs6000 )
         gl_cv_host_cpu_c_abi=powerpc
         ;;

       riscv32 | riscv64 )
         # There are 2 architectures (with variants): rv32* and rv64*.
         AC_COMPILE_IFELSE(
           [AC_LANG_SOURCE(
              [[#if __riscv_xlen == 64
                  int ok;
                #else
                  error fail
                #endif
              ]])],
           [cpu=riscv64],
           [cpu=riscv32])
         # There are 6 ABIs: ilp32, ilp32f, ilp32d, lp64, lp64f, lp64d.
         # Size of 'long' and 'void *':
         AC_COMPILE_IFELSE(
           [AC_LANG_SOURCE(
              [[#if defined __LP64__
                  int ok;
                #else
                  error fail
                #endif
              ]])],
           [main_abi=lp64],
           [main_abi=ilp32])
         # Float ABIs:
         # __riscv_float_abi_double:
         #   'float' and 'double' are passed in floating-point registers.
         # __riscv_float_abi_single:
         #   'float' are passed in floating-point registers.
         # __riscv_float_abi_soft:
         #   No values are passed in floating-point registers.
         AC_COMPILE_IFELSE(
           [AC_LANG_SOURCE(
              [[#if defined __riscv_float_abi_double
                  int ok;
                #else
                  error fail
                #endif
              ]])],
           [float_abi=d],
           [AC_COMPILE_IFELSE(
              [AC_LANG_SOURCE(
                 [[#if defined __riscv_float_abi_single
                     int ok;
                   #else
                     error fail
                   #endif
                 ]])],
              [float_abi=f],
              [float_abi=''])
           ])
         gl_cv_host_cpu_c_abi="${cpu}-${main_abi}${float_abi}"
         ;;

       s390* )
         # On s390x, the C compiler may be generating 64-bit (= s390x) code
         # or 31-bit (= s390) code.
         AC_COMPILE_IFELSE(
           [AC_LANG_SOURCE(
              [[#if defined __LP64__ || defined __s390x__
                  int ok;
                #else
                  error fail
                #endif
              ]])],
           [gl_cv_host_cpu_c_abi=s390x],
           [gl_cv_host_cpu_c_abi=s390])
         ;;

       sparc | sparc64 )
         # UltraSPARCs running Linux have `uname -m` = "sparc64", but the
         # C compiler still generates 32-bit code.
         AC_COMPILE_IFELSE(
           [AC_LANG_SOURCE(
              [[#if defined __sparcv9 || defined __arch64__
                 int ok;
                #else
                 error fail
                #endif
              ]])],
           [gl_cv_host_cpu_c_abi=sparc64],
           [gl_cv_host_cpu_c_abi=sparc])
         ;;

       *)
         gl_cv_host_cpu_c_abi="$host_cpu"
         ;;
     esac
    ])

  dnl In most cases, $HOST_CPU and $HOST_CPU_C_ABI are the same.
  HOST_CPU=`echo "$gl_cv_host_cpu_c_abi" | sed -e 's/-.*//'`
  HOST_CPU_C_ABI="$gl_cv_host_cpu_c_abi"
  AC_SUBST([HOST_CPU])
  AC_SUBST([HOST_CPU_C_ABI])

  # This was
  #   AC_DEFINE_UNQUOTED([__${HOST_CPU}__])
  #   AC_DEFINE_UNQUOTED([__${HOST_CPU_C_ABI}__])
  # earlier, but KAI C++ 3.2d doesn't like this.
  sed -e 's/-/_/g' >> confdefs.h <<EOF
#ifndef __${HOST_CPU}__
#define __${HOST_CPU}__ 1
#endif
#ifndef __${HOST_CPU_C_ABI}__
#define __${HOST_CPU_C_ABI}__ 1
#endif
EOF
  AH_TOP([/* CPU and C ABI indicator */
#ifndef __i386__
#undef __i386__
#endif
#ifndef __x86_64_x32__
#undef __x86_64_x32__
#endif
#ifndef __x86_64__
#undef __x86_64__
#endif
#ifndef __alpha__
#undef __alpha__
#endif
#ifndef __arm__
#undef __arm__
#endif
#ifndef __armhf__
#undef __armhf__
#endif
#ifndef __arm64_ilp32__
#undef __arm64_ilp32__
#endif
#ifndef __arm64__
#undef __arm64__
#endif
#ifndef __hppa__
#undef __hppa__
#endif
#ifndef __hppa64__
#undef __hppa64__
#endif
#ifndef __ia64_ilp32__
#undef __ia64_ilp32__
#endif
#ifndef __ia64__
#undef __ia64__
#endif
#ifndef __loongarch64__
#undef __loongarch64__
#endif
#ifndef __m68k__
#undef __m68k__
#endif
#ifndef __mips__
#undef __mips__
#endif
#ifndef __mipsn32__
#undef __mipsn32__
#endif
#ifndef __mips64__
#undef __mips64__
#endif
#ifndef __powerpc__
#undef __powerpc__
#endif
#ifndef __powerpc64__
#undef __powerpc64__
#endif
#ifndef __powerpc64_elfv2__
#undef __powerpc64_elfv2__
#endif
#ifndef __riscv32__
#undef __riscv32__
#endif
#ifndef __riscv64__
#undef __riscv64__
#endif
#ifndef __riscv32_ilp32__
#undef __riscv32_ilp32__
#endif
#ifndef __riscv32_ilp32f__
#undef __riscv32_ilp32f__
#endif
#ifndef __riscv32_ilp32d__
#undef __riscv32_ilp32d__
#endif
#ifndef __riscv64_ilp32__
#undef __riscv64_ilp32__
#endif
#ifndef __riscv64_ilp32f__
#undef __riscv64_ilp32f__
#endif
#ifndef __riscv64_ilp32d__
#undef __riscv64_ilp32d__
#endif
#ifndef __riscv64_lp64__
#undef __riscv64_lp64__
#endif
#ifndef __riscv64_lp64f__
#undef __riscv64_lp64f__
#endif
#ifndef __riscv64_lp64d__
#undef __riscv64_lp64d__
#endif
#ifndef __s390__
#undef __s390__
#endif
#ifndef __s390x__
#undef __s390x__
#endif
#ifndef __sh__
#undef __sh__
#endif
#ifndef __sparc__
#undef __sparc__
#endif
#ifndef __sparc64__
#undef __sparc64__
#endif
])

])


dnl Sets the HOST_CPU_C_ABI_32BIT variable to 'yes' if the C language ABI
dnl (application binary interface) is a 32-bit one, to 'no' if it is a 64-bit
dnl one, or to 'unknown' if unknown.
dnl This is a simplified variant of gl_HOST_CPU_C_ABI.
AC_DEFUN([gl_HOST_CPU_C_ABI_32BIT],
[
  AC_REQUIRE([AC_CANONICAL_HOST])
  AC_CACHE_CHECK([32-bit host C ABI], [gl_cv_host_cpu_c_abi_32bit],
    [if test -n "$gl_cv_host_cpu_c_abi"; then
       case "$gl_cv_host_cpu_c_abi" in
         i386 | x86_64-x32 | arm | armhf | arm64-ilp32 | hppa | ia64-ilp32 | mips | mipsn32 | powerpc | riscv*-ilp32* | s390 | sparc)
           gl_cv_host_cpu_c_abi_32bit=yes ;;
         x86_64 | alpha | arm64 | hppa64 | ia64 | mips64 | powerpc64 | powerpc64-elfv2 | riscv*-lp64* | s390x | sparc64 )
           gl_cv_host_cpu_c_abi_32bit=no ;;
         *)
           gl_cv_host_cpu_c_abi_32bit=unknown ;;
       esac
     else
       case "$host_cpu" in

         # CPUs that only support a 32-bit ABI.
         arc \
         | bfin \
         | cris* \
         | csky \
         | epiphany \
         | ft32 \
         | h8300 \
         | m68k \
         | microblaze | microblazeel \
         | nds32 | nds32le | nds32be \
         | nios2 | nios2eb | nios2el \
         | or1k* \
         | or32 \
         | sh | sh[1234] | sh[1234]e[lb] \
         | tic6x \
         | xtensa* )
           gl_cv_host_cpu_c_abi_32bit=yes
           ;;

         # CPUs that only support a 64-bit ABI.
changequote(,)dnl
         alpha | alphaev[4-8] | alphaev56 | alphapca5[67] | alphaev6[78] \
         | mmix )
changequote([,])dnl
           gl_cv_host_cpu_c_abi_32bit=no
           ;;

changequote(,)dnl
         i[34567]86 )
changequote([,])dnl
           gl_cv_host_cpu_c_abi_32bit=yes
           ;;

         x86_64 )
           # On x86_64 systems, the C compiler may be generating code in one of
           # these ABIs:
           # - 64-bit instruction set, 64-bit pointers, 64-bit 'long': x86_64.
           # - 64-bit instruction set, 64-bit pointers, 32-bit 'long': x86_64
           #   with native Windows (mingw, MSVC).
           # - 64-bit instruction set, 32-bit pointers, 32-bit 'long': x86_64-x32.
           # - 32-bit instruction set, 32-bit pointers, 32-bit 'long': i386.
           AC_COMPILE_IFELSE(
             [AC_LANG_SOURCE(
                [[#if (defined __x86_64__ || defined __amd64__ \
                       || defined _M_X64 || defined _M_AMD64) \
                      && !(defined __ILP32__ || defined _ILP32)
                   int ok;
                  #else
                   error fail
                  #endif
                ]])],
             [gl_cv_host_cpu_c_abi_32bit=no],
             [gl_cv_host_cpu_c_abi_32bit=yes])
           ;;

         arm* | aarch64 )
           # Assume arm with EABI.
           # On arm64 systems, the C compiler may be generating code in one of
           # these ABIs:
           # - aarch64 instruction set, 64-bit pointers, 64-bit 'long': arm64.
           # - aarch64 instruction set, 32-bit pointers, 32-bit 'long': arm64-ilp32.
           # - 32-bit instruction set, 32-bit pointers, 32-bit 'long': arm or armhf.
           AC_COMPILE_IFELSE(
             [AC_LANG_SOURCE(
                [[#if defined __aarch64__ && !(defined __ILP32__ || defined _ILP32)
                   int ok;
                  #else
                   error fail
                  #endif
                ]])],
             [gl_cv_host_cpu_c_abi_32bit=no],
             [gl_cv_host_cpu_c_abi_32bit=yes])
           ;;

         hppa1.0 | hppa1.1 | hppa2.0* | hppa64 )
           # On hppa, the C compiler may be generating 32-bit code or 64-bit
           # code. In the latter case, it defines _LP64 and __LP64__.
           AC_COMPILE_IFELSE(
             [AC_LANG_SOURCE(
                [[#ifdef __LP64__
                   int ok;
                  #else
                   error fail
                  #endif
                ]])],
             [gl_cv_host_cpu_c_abi_32bit=no],
             [gl_cv_host_cpu_c_abi_32bit=yes])
           ;;

         ia64* )
           # On ia64 on HP-UX, the C compiler may be generating 64-bit code or
           # 32-bit code. In the latter case, it defines _ILP32.
           AC_COMPILE_IFELSE(
             [AC_LANG_SOURCE(
                [[#ifdef _ILP32
                   int ok;
                  #else
                   error fail
                  #endif
                ]])],
             [gl_cv_host_cpu_c_abi_32bit=yes],
             [gl_cv_host_cpu_c_abi_32bit=no])
           ;;

         mips* )
           # We should also check for (_MIPS_SZPTR == 64), but gcc keeps this
           # at 32.
           AC_COMPILE_IFELSE(
             [AC_LANG_SOURCE(
                [[#if defined _MIPS_SZLONG && (_MIPS_SZLONG == 64)
                   int ok;
                  #else
                   error fail
                  #endif
                ]])],
             [gl_cv_host_cpu_c_abi_32bit=no],
             [gl_cv_host_cpu_c_abi_32bit=yes])
           ;;

         powerpc* )
           # Different ABIs are in use on AIX vs. Mac OS X vs. Linux,*BSD.
           # No need to distinguish them here; the caller may distinguish
           # them based on the OS.
           # On powerpc64 systems, the C compiler may still be generating
           # 32-bit code. And on powerpc-ibm-aix systems, the C compiler may
           # be generating 64-bit code.
           AC_COMPILE_IFELSE(
             [AC_LANG_SOURCE(
                [[#if defined __powerpc64__ || defined __LP64__
                   int ok;
                  #else
                   error fail
                  #endif
                ]])],
             [gl_cv_host_cpu_c_abi_32bit=no],
             [gl_cv_host_cpu_c_abi_32bit=yes])
           ;;

         rs6000 )
           gl_cv_host_cpu_c_abi_32bit=yes
           ;;

         riscv32 | riscv64 )
           # There are 6 ABIs: ilp32, ilp32f, ilp32d, lp64, lp64f, lp64d.
           # Size of 'long' and 'void *':
           AC_COMPILE_IFELSE(
             [AC_LANG_SOURCE(
                [[#if defined __LP64__
                    int ok;
                  #else
                    error fail
                  #endif
                ]])],
             [gl_cv_host_cpu_c_abi_32bit=no],
             [gl_cv_host_cpu_c_abi_32bit=yes])
           ;;

         s390* )
           # On s390x, the C compiler may be generating 64-bit (= s390x) code
           # or 31-bit (= s390) code.
           AC_COMPILE_IFELSE(
             [AC_LANG_SOURCE(
                [[#if defined __LP64__ || defined __s390x__
                    int ok;
                  #else
                    error fail
                  #endif
                ]])],
             [gl_cv_host_cpu_c_abi_32bit=no],
             [gl_cv_host_cpu_c_abi_32bit=yes])
           ;;

         sparc | sparc64 )
           # UltraSPARCs running Linux have `uname -m` = "sparc64", but the
           # C compiler still generates 32-bit code.
           AC_COMPILE_IFELSE(
             [AC_LANG_SOURCE(
                [[#if defined __sparcv9 || defined __arch64__
                   int ok;
                  #else
                   error fail
                  #endif
                ]])],
             [gl_cv_host_cpu_c_abi_32bit=no],
             [gl_cv_host_cpu_c_abi_32bit=yes])
           ;;

         *)
           gl_cv_host_cpu_c_abi_32bit=unknown
           ;;
       esac
     fi
    ])

  HOST_CPU_C_ABI_32BIT="$gl_cv_host_cpu_c_abi_32bit"
])
