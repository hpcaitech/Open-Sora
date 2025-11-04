/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* How many MiB of RAM to assume if the real amount cannot be determined. */
#define ASSUME_RAM 128

/* Define to 1 if translation of program messages to the user's native
   language is requested. */
/* #undef ENABLE_NLS */

/* Define to 1 if bswap_16 is available. */
/* #undef HAVE_BSWAP_16 */

/* Define to 1 if bswap_32 is available. */
/* #undef HAVE_BSWAP_32 */

/* Define to 1 if bswap_64 is available. */
/* #undef HAVE_BSWAP_64 */

/* Define to 1 if you have the <byteswap.h> header file. */
/* #undef HAVE_BYTESWAP_H */

/* Define to 1 if Capsicum is available. */
/* #undef HAVE_CAPSICUM */

/* Define to 1 if the system has the type `CC_SHA256_CTX'. */
/* #undef HAVE_CC_SHA256_CTX */

/* Define to 1 if you have the `CC_SHA256_Init' function. */
/* #undef HAVE_CC_SHA256_INIT */

/* Define to 1 if you have the Mac OS X function
   CFLocaleCopyPreferredLanguages in the CoreFoundation framework. */
#define HAVE_CFLOCALECOPYPREFERREDLANGUAGES 1

/* Define to 1 if you have the Mac OS X function CFPreferencesCopyAppValue in
   the CoreFoundation framework. */
#define HAVE_CFPREFERENCESCOPYAPPVALUE 1

/* Define to 1 if crc32 integrity check is enabled. */
#define HAVE_CHECK_CRC32 1

/* Define to 1 if crc64 integrity check is enabled. */
#define HAVE_CHECK_CRC64 1

/* Define to 1 if sha256 integrity check is enabled. */
#define HAVE_CHECK_SHA256 1

/* Define to 1 if you have the `clock_gettime' function. */
#define HAVE_CLOCK_GETTIME 1

/* Define to 1 if `CLOCK_MONOTONIC' is declared in <time.h>. */
#define HAVE_CLOCK_MONOTONIC 1

/* Define to 1 if you have the <CommonCrypto/CommonDigest.h> header file. */
/* #undef HAVE_COMMONCRYPTO_COMMONDIGEST_H */

/* Define to 1 if you have the <cpuid.h> header file. */
#define HAVE_CPUID_H 1

/* Define if the GNU dcgettext() function is already present or preinstalled.
   */
/* #undef HAVE_DCGETTEXT */

/* Define to 1 if any of HAVE_DECODER_foo have been defined. */
#define HAVE_DECODERS 1

/* Define to 1 if arm decoder is enabled. */
#define HAVE_DECODER_ARM 1

/* Define to 1 if arm64 decoder is enabled. */
#define HAVE_DECODER_ARM64 1

/* Define to 1 if armthumb decoder is enabled. */
#define HAVE_DECODER_ARMTHUMB 1

/* Define to 1 if delta decoder is enabled. */
#define HAVE_DECODER_DELTA 1

/* Define to 1 if ia64 decoder is enabled. */
#define HAVE_DECODER_IA64 1

/* Define to 1 if lzma1 decoder is enabled. */
#define HAVE_DECODER_LZMA1 1

/* Define to 1 if lzma2 decoder is enabled. */
#define HAVE_DECODER_LZMA2 1

/* Define to 1 if powerpc decoder is enabled. */
#define HAVE_DECODER_POWERPC 1

/* Define to 1 if sparc decoder is enabled. */
#define HAVE_DECODER_SPARC 1

/* Define to 1 if x86 decoder is enabled. */
#define HAVE_DECODER_X86 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if any of HAVE_ENCODER_foo have been defined. */
#define HAVE_ENCODERS 1

/* Define to 1 if arm encoder is enabled. */
#define HAVE_ENCODER_ARM 1

/* Define to 1 if arm64 encoder is enabled. */
#define HAVE_ENCODER_ARM64 1

/* Define to 1 if armthumb encoder is enabled. */
#define HAVE_ENCODER_ARMTHUMB 1

/* Define to 1 if delta encoder is enabled. */
#define HAVE_ENCODER_DELTA 1

/* Define to 1 if ia64 encoder is enabled. */
#define HAVE_ENCODER_IA64 1

/* Define to 1 if lzma1 encoder is enabled. */
#define HAVE_ENCODER_LZMA1 1

/* Define to 1 if lzma2 encoder is enabled. */
#define HAVE_ENCODER_LZMA2 1

/* Define to 1 if powerpc encoder is enabled. */
#define HAVE_ENCODER_POWERPC 1

/* Define to 1 if sparc encoder is enabled. */
#define HAVE_ENCODER_SPARC 1

/* Define to 1 if x86 encoder is enabled. */
#define HAVE_ENCODER_X86 1

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if __attribute__((__constructor__)) is supported for functions.
   */
#define HAVE_FUNC_ATTRIBUTE_CONSTRUCTOR 1

/* Define to 1 if you have the `futimens' function. */
#define HAVE_FUTIMENS 1

/* Define to 1 if you have the `futimes' function. */
/* #undef HAVE_FUTIMES */

/* Define to 1 if you have the `futimesat' function. */
/* #undef HAVE_FUTIMESAT */

/* Define to 1 if you have the <getopt.h> header file. */
#define HAVE_GETOPT_H 1

/* Define to 1 if you have the `getopt_long' function. */
#define HAVE_GETOPT_LONG 1

/* Define if the GNU gettext() function is already present or preinstalled. */
/* #undef HAVE_GETTEXT */

/* Define if you have the iconv() function and it works. */
#define HAVE_ICONV 1

/* Define to 1 if you have the <immintrin.h> header file. */
#define HAVE_IMMINTRIN_H 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <limits.h> header file. */
#define HAVE_LIMITS_H 1

/* Define to 1 if .lz (lzip) decompression support is enabled. */
#define HAVE_LZIP_DECODER 1

/* Define to 1 if mbrtowc and mbstate_t are properly declared. */
#define HAVE_MBRTOWC 1

/* Define to 1 to enable bt2 match finder. */
#define HAVE_MF_BT2 1

/* Define to 1 to enable bt3 match finder. */
#define HAVE_MF_BT3 1

/* Define to 1 to enable bt4 match finder. */
#define HAVE_MF_BT4 1

/* Define to 1 to enable hc3 match finder. */
#define HAVE_MF_HC3 1

/* Define to 1 to enable hc4 match finder. */
#define HAVE_MF_HC4 1

/* Define to 1 if you have the <minix/config.h> header file. */
/* #undef HAVE_MINIX_CONFIG_H */

/* Define to 1 if getopt.h declares extern int optreset. */
#define HAVE_OPTRESET 1

/* Define to 1 if you have the `pledge' function. */
/* #undef HAVE_PLEDGE */

/* Define to 1 if you have the `posix_fadvise' function. */
/* #undef HAVE_POSIX_FADVISE */

/* Define to 1 if `program_invocation_name' is declared in <errno.h>. */
/* #undef HAVE_PROGRAM_INVOCATION_NAME */

/* Define to 1 if you have the `pthread_condattr_setclock' function. */
/* #undef HAVE_PTHREAD_CONDATTR_SETCLOCK */

/* Have PTHREAD_PRIO_INHERIT. */
#define HAVE_PTHREAD_PRIO_INHERIT 1

/* Define to 1 if you have the `SHA256Init' function. */
/* #undef HAVE_SHA256INIT */

/* Define to 1 if the system has the type `SHA256_CTX'. */
/* #undef HAVE_SHA256_CTX */

/* Define to 1 if you have the <sha256.h> header file. */
/* #undef HAVE_SHA256_H */

/* Define to 1 if you have the `SHA256_Init' function. */
/* #undef HAVE_SHA256_INIT */

/* Define to 1 if the system has the type `SHA2_CTX'. */
/* #undef HAVE_SHA2_CTX */

/* Define to 1 if you have the <sha2.h> header file. */
/* #undef HAVE_SHA2_H */

/* Define to 1 if optimizing for size. */
/* #undef HAVE_SMALL */

/* Define to 1 if stdbool.h conforms to C99. */
#define HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if `st_atimensec' is a member of `struct stat'. */
/* #undef HAVE_STRUCT_STAT_ST_ATIMENSEC */

/* Define to 1 if `st_atimespec.tv_nsec' is a member of `struct stat'. */
#define HAVE_STRUCT_STAT_ST_ATIMESPEC_TV_NSEC 1

/* Define to 1 if `st_atim.st__tim.tv_nsec' is a member of `struct stat'. */
/* #undef HAVE_STRUCT_STAT_ST_ATIM_ST__TIM_TV_NSEC */

/* Define to 1 if `st_atim.tv_nsec' is a member of `struct stat'. */
/* #undef HAVE_STRUCT_STAT_ST_ATIM_TV_NSEC */

/* Define to 1 if `st_uatime' is a member of `struct stat'. */
/* #undef HAVE_STRUCT_STAT_ST_UATIME */

/* Define to 1 to if GNU/Linux-specific details are unconditionally wanted for
   symbol versioning. Define to 2 to if these are wanted only if also PIC is
   defined (allows building both shared and static liblzma at the same time
   with Libtool if neither --with-pic nor --without-pic is used). This define
   must be used together with liblzma_linux.map. */
/* #undef HAVE_SYMBOL_VERSIONS_LINUX */

/* Define to 1 if you have the <sys/byteorder.h> header file. */
/* #undef HAVE_SYS_BYTEORDER_H */

/* Define to 1 if you have the <sys/capsicum.h> header file. */
/* #undef HAVE_SYS_CAPSICUM_H */

/* Define to 1 if you have the <sys/endian.h> header file. */
/* #undef HAVE_SYS_ENDIAN_H */

/* Define to 1 if you have the <sys/param.h> header file. */
#define HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if the system has the type `uintptr_t'. */
#define HAVE_UINTPTR_T 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if _mm_set_epi64x and _mm_clmulepi64_si128 are usable. See
   configure.ac for details. */
#define HAVE_USABLE_CLMUL 1

/* Define to 1 if you have the `utime' function. */
/* #undef HAVE_UTIME */

/* Define to 1 if you have the `utimes' function. */
/* #undef HAVE_UTIMES */

/* Define to 1 or 0, depending whether the compiler supports simple visibility
   declarations. */
#define HAVE_VISIBILITY 1

/* Define to 1 if you have the <wchar.h> header file. */
#define HAVE_WCHAR_H 1

/* Define to 1 if you have the `wcwidth' function. */
#define HAVE_WCWIDTH 1

/* Define to 1 if the system has the type `_Bool'. */
#define HAVE__BOOL 1

/* Define to 1 if you have the `_futime' function. */
/* #undef HAVE__FUTIME */

/* Define to 1 if _mm_movemask_epi8 is available. */
#define HAVE__MM_MOVEMASK_EPI8 1

/* Define to 1 if the GNU C extension __builtin_assume_aligned is supported.
   */
#define HAVE___BUILTIN_ASSUME_ALIGNED 1

/* Define to 1 if the GNU C extensions __builtin_bswap16/32/64 are supported.
   */
#define HAVE___BUILTIN_BSWAPXX 1

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Define to 1 when using POSIX threads (pthreads). */
#define MYTHREAD_POSIX 1

/* Define to 1 when using Windows Vista compatible threads. This uses features
   that are not available on Windows XP. */
/* #undef MYTHREAD_VISTA */

/* Define to 1 when using Windows 95 (and thus XP) compatible threads. This
   avoids use of features that were added in Windows Vista. */
/* #undef MYTHREAD_WIN95 */

/* Define to 1 to disable debugging code. */
#define NDEBUG 1

/* Name of package */
#define PACKAGE "xz"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "xz@tukaani.org"

/* Define to the full name of this package. */
#define PACKAGE_NAME "XZ Utils"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "XZ Utils 5.4.5"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "xz"

/* Define to the home page for this package. */
#define PACKAGE_URL "https://tukaani.org/xz/"

/* Define to the version of this package. */
#define PACKAGE_VERSION "5.4.5"

/* Define to necessary symbol if this constant uses a non-standard name on
   your system. */
/* #undef PTHREAD_CREATE_JOINABLE */

/* The size of `size_t', as computed by sizeof. */
#define SIZEOF_SIZE_T 8

/* Define to 1 if all of the C90 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
#define STDC_HEADERS 1

/* Define to 1 if the number of available CPU cores can be detected with
   cpuset(2). */
/* #undef TUKLIB_CPUCORES_CPUSET */

/* Define to 1 if the number of available CPU cores can be detected with
   pstat_getdynamic(). */
/* #undef TUKLIB_CPUCORES_PSTAT_GETDYNAMIC */

/* Define to 1 if the number of available CPU cores can be detected with
   sched_getaffinity() */
/* #undef TUKLIB_CPUCORES_SCHED_GETAFFINITY */

/* Define to 1 if the number of available CPU cores can be detected with
   sysconf(_SC_NPROCESSORS_ONLN) or sysconf(_SC_NPROC_ONLN). */
/* #undef TUKLIB_CPUCORES_SYSCONF */

/* Define to 1 if the number of available CPU cores can be detected with
   sysctl(). */
#define TUKLIB_CPUCORES_SYSCTL 1

/* Define to 1 if the system supports fast unaligned access to 16-bit, 32-bit,
   and 64-bit integers. */
#define TUKLIB_FAST_UNALIGNED_ACCESS 1

/* Define to 1 if the amount of physical memory can be detected with
   _system_configuration.physmem. */
/* #undef TUKLIB_PHYSMEM_AIX */

/* Define to 1 if the amount of physical memory can be detected with
   getinvent_r(). */
/* #undef TUKLIB_PHYSMEM_GETINVENT_R */

/* Define to 1 if the amount of physical memory can be detected with
   getsysinfo(). */
/* #undef TUKLIB_PHYSMEM_GETSYSINFO */

/* Define to 1 if the amount of physical memory can be detected with
   pstat_getstatic(). */
/* #undef TUKLIB_PHYSMEM_PSTAT_GETSTATIC */

/* Define to 1 if the amount of physical memory can be detected with
   sysconf(_SC_PAGESIZE) and sysconf(_SC_PHYS_PAGES). */
#define TUKLIB_PHYSMEM_SYSCONF 1

/* Define to 1 if the amount of physical memory can be detected with sysctl().
   */
/* #undef TUKLIB_PHYSMEM_SYSCTL */

/* Define to 1 if the amount of physical memory can be detected with Linux
   sysinfo(). */
/* #undef TUKLIB_PHYSMEM_SYSINFO */

/* Define to 1 to use unsafe type punning, e.g. char *x = ...; *(int *)x =
   123; which violates strict aliasing rules and thus is undefined behavior
   and might result in broken code. */
/* #undef TUKLIB_USE_UNSAFE_TYPE_PUNNING */

/* Enable extensions on AIX 3, Interix.  */
#ifndef _ALL_SOURCE
# define _ALL_SOURCE 1
#endif
/* Enable general extensions on macOS.  */
#ifndef _DARWIN_C_SOURCE
# define _DARWIN_C_SOURCE 1
#endif
/* Enable general extensions on Solaris.  */
#ifndef __EXTENSIONS__
# define __EXTENSIONS__ 1
#endif
/* Enable GNU extensions on systems that have them.  */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif
/* Enable X/Open compliant socket functions that do not require linking
   with -lxnet on HP-UX 11.11.  */
#ifndef _HPUX_ALT_XOPEN_SOCKET_API
# define _HPUX_ALT_XOPEN_SOCKET_API 1
#endif
/* Identify the host operating system as Minix.
   This macro does not affect the system headers' behavior.
   A future release of Autoconf may stop defining this macro.  */
#ifndef _MINIX
/* # undef _MINIX */
#endif
/* Enable general extensions on NetBSD.
   Enable NetBSD compatibility extensions on Minix.  */
#ifndef _NETBSD_SOURCE
# define _NETBSD_SOURCE 1
#endif
/* Enable OpenBSD compatibility extensions on NetBSD.
   Oddly enough, this does nothing on OpenBSD.  */
#ifndef _OPENBSD_SOURCE
# define _OPENBSD_SOURCE 1
#endif
/* Define to 1 if needed for POSIX-compatible behavior.  */
#ifndef _POSIX_SOURCE
/* # undef _POSIX_SOURCE */
#endif
/* Define to 2 if needed for POSIX-compatible behavior.  */
#ifndef _POSIX_1_SOURCE
/* # undef _POSIX_1_SOURCE */
#endif
/* Enable POSIX-compatible threading on Solaris.  */
#ifndef _POSIX_PTHREAD_SEMANTICS
# define _POSIX_PTHREAD_SEMANTICS 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-5:2014.  */
#ifndef __STDC_WANT_IEC_60559_ATTRIBS_EXT__
# define __STDC_WANT_IEC_60559_ATTRIBS_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-1:2014.  */
#ifndef __STDC_WANT_IEC_60559_BFP_EXT__
# define __STDC_WANT_IEC_60559_BFP_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-2:2015.  */
#ifndef __STDC_WANT_IEC_60559_DFP_EXT__
# define __STDC_WANT_IEC_60559_DFP_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-4:2015.  */
#ifndef __STDC_WANT_IEC_60559_FUNCS_EXT__
# define __STDC_WANT_IEC_60559_FUNCS_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-3:2015.  */
#ifndef __STDC_WANT_IEC_60559_TYPES_EXT__
# define __STDC_WANT_IEC_60559_TYPES_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TR 24731-2:2010.  */
#ifndef __STDC_WANT_LIB_EXT2__
# define __STDC_WANT_LIB_EXT2__ 1
#endif
/* Enable extensions specified by ISO/IEC 24747:2009.  */
#ifndef __STDC_WANT_MATH_SPEC_FUNCS__
# define __STDC_WANT_MATH_SPEC_FUNCS__ 1
#endif
/* Enable extensions on HP NonStop.  */
#ifndef _TANDEM_SOURCE
# define _TANDEM_SOURCE 1
#endif
/* Enable X/Open extensions.  Define to 500 only if necessary
   to make mbstate_t available.  */
#ifndef _XOPEN_SOURCE
/* # undef _XOPEN_SOURCE */
#endif


/* Version number of package */
#define VERSION "5.4.5"

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif

/* Number of bits in a file offset, on hosts where this is settable. */
/* #undef _FILE_OFFSET_BITS */

/* Define for large files, on AIX-style hosts. */
/* #undef _LARGE_FILES */

/* Define for Solaris 2.5.1 so the uint32_t typedef from <sys/synch.h>,
   <pthread.h>, or <semaphore.h> is not used. If the typedef were allowed, the
   #define below would cause a syntax error. */
/* #undef _UINT32_T */

/* Define for Solaris 2.5.1 so the uint64_t typedef from <sys/synch.h>,
   <pthread.h>, or <semaphore.h> is not used. If the typedef were allowed, the
   #define below would cause a syntax error. */
/* #undef _UINT64_T */

/* Define for Solaris 2.5.1 so the uint8_t typedef from <sys/synch.h>,
   <pthread.h>, or <semaphore.h> is not used. If the typedef were allowed, the
   #define below would cause a syntax error. */
/* #undef _UINT8_T */

/* Define to rpl_ if the getopt replacement functions and variables should be
   used. */
/* #undef __GETOPT_PREFIX */

/* Define to the type of a signed integer type of width exactly 32 bits if
   such a type exists and the standard includes do not define it. */
/* #undef int32_t */

/* Define to the type of a signed integer type of width exactly 64 bits if
   such a type exists and the standard includes do not define it. */
/* #undef int64_t */

/* Define to the type of an unsigned integer type of width exactly 16 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint16_t */

/* Define to the type of an unsigned integer type of width exactly 32 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint32_t */

/* Define to the type of an unsigned integer type of width exactly 64 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint64_t */

/* Define to the type of an unsigned integer type of width exactly 8 bits if
   such a type exists and the standard includes do not define it. */
/* #undef uint8_t */

/* Define to the type of an unsigned integer type wide enough to hold a
   pointer, if such a type exists, and if the system does not define it. */
/* #undef uintptr_t */
