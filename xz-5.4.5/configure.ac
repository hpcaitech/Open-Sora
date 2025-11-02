#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

###############################################################################
#
# Author: Lasse Collin
#
# This file has been put into the public domain.
# You can do whatever you want with this file.
#
###############################################################################

# NOTE: Don't add useless checks. autoscan detects this and that, but don't
# let it confuse you. For example, we don't care about checking for behavior
# of malloc(), stat(), or lstat(), since we don't use those functions in
# a way that would cause the problems the autoconf macros check.

AC_PREREQ([2.69])

AC_INIT([XZ Utils], m4_esyscmd([/bin/sh build-aux/version.sh]),
	[xz@tukaani.org], [xz], [https://tukaani.org/xz/])
AC_CONFIG_SRCDIR([src/liblzma/common/common.h])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])

echo
echo "$PACKAGE_STRING"

echo
echo "System type:"
# This is needed to know if assembler optimizations can be used.
AC_CANONICAL_HOST

# We do some special things on Windows (32-bit or 64-bit) builds.
case $host_os in
	mingw* | cygwin | msys) is_w32=yes ;;
	*)                      is_w32=no ;;
esac
AM_CONDITIONAL([COND_W32], [test "$is_w32" = yes])

# We need to use $EXEEXT with $(LN_S) when creating symlinks to
# executables. Cygwin is an exception to this, since it is recommended
# that symlinks don't have the .exe suffix. To make this work, we
# define LN_EXEEXT.
#
# MSYS2 is treated the same way as Cygwin. It uses plain "msys" like
# the original MSYS when building MSYS/MSYS2-binaries. Hopefully this
# doesn't break things for the original MSYS developers. Note that this
# doesn't affect normal MSYS/MSYS2 users building non-MSYS/MSYS2 binaries
# since in that case the $host_os is usually mingw32.
case $host_os in
	cygwin | msys)  LN_EXEEXT= ;;
	*)              LN_EXEEXT='$(EXEEXT)' ;;
esac
AC_SUBST([LN_EXEEXT])

echo
echo "Configure options:"
AM_CFLAGS=


#############
# Debugging #
#############

AC_MSG_CHECKING([if debugging code should be compiled])
AC_ARG_ENABLE([debug], AS_HELP_STRING([--enable-debug], [Enable debugging code.]),
	[], enable_debug=no)
if test "x$enable_debug" = xyes; then
	AC_MSG_RESULT([yes])
else
	AC_DEFINE([NDEBUG], [1], [Define to 1 to disable debugging code.])
	AC_MSG_RESULT([no])
fi


###########
# Filters #
###########

m4_define([SUPPORTED_FILTERS], [lzma1,lzma2,delta,x86,powerpc,ia64,arm,armthumb,arm64,sparc])dnl
m4_define([SIMPLE_FILTERS], [x86,powerpc,ia64,arm,armthumb,arm64,sparc])
m4_define([LZ_FILTERS], [lzma1,lzma2])

m4_foreach([NAME], [SUPPORTED_FILTERS],
[enable_filter_[]NAME=no
enable_encoder_[]NAME=no
enable_decoder_[]NAME=no
])dnl

AC_MSG_CHECKING([which encoders to build])
AC_ARG_ENABLE([encoders], AS_HELP_STRING([--enable-encoders=LIST],
		[Comma-separated list of encoders to build. Default=all.
		Available encoders:]
			m4_translit(m4_defn([SUPPORTED_FILTERS]), [,], [ ])),
	[], [enable_encoders=SUPPORTED_FILTERS])
enable_encoders=`echo "$enable_encoders" | sed 's/,/ /g'`
if test "x$enable_encoders" = xno || test "x$enable_encoders" = x; then
	enable_encoders=no
	AC_MSG_RESULT([(none)])
else
	for arg in $enable_encoders
	do
		case $arg in m4_foreach([NAME], [SUPPORTED_FILTERS], [
			NAME)
				enable_filter_[]NAME=yes
				enable_encoder_[]NAME=yes
				AC_DEFINE(HAVE_ENCODER_[]m4_toupper(NAME), [1],
				[Define to 1 if] NAME [encoder is enabled.])
				;;])
			*)
				AC_MSG_RESULT([])
				AC_MSG_ERROR([unknown filter: $arg])
				;;
		esac
	done
	AC_DEFINE([HAVE_ENCODERS], [1],
		[Define to 1 if any of HAVE_ENCODER_foo have been defined.])
	AC_MSG_RESULT([$enable_encoders])
fi

AC_MSG_CHECKING([which decoders to build])
AC_ARG_ENABLE([decoders], AS_HELP_STRING([--enable-decoders=LIST],
		[Comma-separated list of decoders to build. Default=all.
		Available decoders are the same as available encoders.]),
	[], [enable_decoders=SUPPORTED_FILTERS])
enable_decoders=`echo "$enable_decoders" | sed 's/,/ /g'`
if test "x$enable_decoders" = xno || test "x$enable_decoders" = x; then
	enable_decoders=no
	AC_MSG_RESULT([(none)])
else
	for arg in $enable_decoders
	do
		case $arg in m4_foreach([NAME], [SUPPORTED_FILTERS], [
			NAME)
				enable_filter_[]NAME=yes
				enable_decoder_[]NAME=yes
				AC_DEFINE(HAVE_DECODER_[]m4_toupper(NAME), [1],
				[Define to 1 if] NAME [decoder is enabled.])
				;;])
			*)
				AC_MSG_RESULT([])
				AC_MSG_ERROR([unknown filter: $arg])
				;;
		esac
	done
	AC_DEFINE([HAVE_DECODERS], [1],
		[Define to 1 if any of HAVE_DECODER_foo have been defined.])
	AC_MSG_RESULT([$enable_decoders])
fi

if test "x$enable_encoder_lzma2$enable_encoder_lzma1" = xyesno \
		|| test "x$enable_decoder_lzma2$enable_decoder_lzma1" = xyesno; then
	AC_MSG_ERROR([LZMA2 requires that LZMA1 is also enabled.])
fi

AM_CONDITIONAL(COND_MAIN_ENCODER, test "x$enable_encoders" != xno)
AM_CONDITIONAL(COND_MAIN_DECODER, test "x$enable_decoders" != xno)

m4_foreach([NAME], [SUPPORTED_FILTERS],
[AM_CONDITIONAL(COND_FILTER_[]m4_toupper(NAME), test "x$enable_filter_[]NAME" = xyes)
AM_CONDITIONAL(COND_ENCODER_[]m4_toupper(NAME), test "x$enable_encoder_[]NAME" = xyes)
AM_CONDITIONAL(COND_DECODER_[]m4_toupper(NAME), test "x$enable_decoder_[]NAME" = xyes)
])dnl

# The so called "simple filters" share common code.
enable_filter_simple=no
enable_encoder_simple=no
enable_decoder_simple=no
m4_foreach([NAME], [SIMPLE_FILTERS],
[test "x$enable_filter_[]NAME" = xyes && enable_filter_simple=yes
test "x$enable_encoder_[]NAME" = xyes && enable_encoder_simple=yes
test "x$enable_decoder_[]NAME" = xyes && enable_decoder_simple=yes
])dnl
AM_CONDITIONAL(COND_FILTER_SIMPLE, test "x$enable_filter_simple" = xyes)
AM_CONDITIONAL(COND_ENCODER_SIMPLE, test "x$enable_encoder_simple" = xyes)
AM_CONDITIONAL(COND_DECODER_SIMPLE, test "x$enable_decoder_simple" = xyes)

# LZ-based filters share common code.
enable_filter_lz=no
enable_encoder_lz=no
enable_decoder_lz=no
m4_foreach([NAME], [LZ_FILTERS],
[test "x$enable_filter_[]NAME" = xyes && enable_filter_lz=yes
test "x$enable_encoder_[]NAME" = xyes && enable_encoder_lz=yes
test "x$enable_decoder_[]NAME" = xyes && enable_decoder_lz=yes
])dnl
AM_CONDITIONAL(COND_FILTER_LZ, test "x$enable_filter_lz" = xyes)
AM_CONDITIONAL(COND_ENCODER_LZ, test "x$enable_encoder_lz" = xyes)
AM_CONDITIONAL(COND_DECODER_LZ, test "x$enable_decoder_lz" = xyes)


#################
# Match finders #
#################

m4_define([SUPPORTED_MATCH_FINDERS], [hc3,hc4,bt2,bt3,bt4])

m4_foreach([NAME], [SUPPORTED_MATCH_FINDERS],
[enable_match_finder_[]NAME=no
])

AC_MSG_CHECKING([which match finders to build])
AC_ARG_ENABLE([match-finders], AS_HELP_STRING([--enable-match-finders=LIST],
		[Comma-separated list of match finders to build. Default=all.
		At least one match finder is required for encoding with
		the LZMA1 and LZMA2 filters. Available match finders:]
		m4_translit(m4_defn([SUPPORTED_MATCH_FINDERS]), [,], [ ])), [],
	[enable_match_finders=SUPPORTED_MATCH_FINDERS])
enable_match_finders=`echo "$enable_match_finders" | sed 's/,/ /g'`
if test "x$enable_encoder_lz" = xyes ; then
	if test -z "$enable_match_finders"; then
		AC_MSG_ERROR([At least one match finder is required for an LZ-based encoder.])
	fi

	for arg in $enable_match_finders
		do
		case $arg in m4_foreach([NAME], [SUPPORTED_MATCH_FINDERS], [
			NAME)
				enable_match_finder_[]NAME=yes
				AC_DEFINE(HAVE_MF_[]m4_toupper(NAME), [1],
				[Define to 1 to enable] NAME [match finder.])
				;;])
			*)
				AC_MSG_RESULT([])
				AC_MSG_ERROR([unknown match finder: $arg])
				;;
		esac
	done
	AC_MSG_RESULT([$enable_match_finders])
else
	AC_MSG_RESULT([(none because not building any LZ-based encoder)])
fi


####################
# Integrity checks #
####################

m4_define([SUPPORTED_CHECKS], [crc32,crc64,sha256])

m4_foreach([NAME], [SUPPORTED_CHECKS],
[enable_check_[]NAME=no
])dnl

AC_MSG_CHECKING([which integrity checks to build])
AC_ARG_ENABLE([checks], AS_HELP_STRING([--enable-checks=LIST],
		[Comma-separated list of integrity checks to build.
		Default=all. Available integrity checks:]
		m4_translit(m4_defn([SUPPORTED_CHECKS]), [,], [ ])),
	[], [enable_checks=SUPPORTED_CHECKS])
enable_checks=`echo "$enable_checks" | sed 's/,/ /g'`
if test "x$enable_checks" = xno || test "x$enable_checks" = x; then
	AC_MSG_RESULT([(none)])
else
	for arg in $enable_checks
	do
		case $arg in m4_foreach([NAME], [SUPPORTED_CHECKS], [
			NAME)
				enable_check_[]NAME=yes
				AC_DEFINE(HAVE_CHECK_[]m4_toupper(NAME), [1],
				[Define to 1 if] NAME
				[integrity check is enabled.])
				;;])
			*)
				AC_MSG_RESULT([])
				AC_MSG_ERROR([unknown integrity check: $arg])
				;;
		esac
	done
	AC_MSG_RESULT([$enable_checks])
fi
if test "x$enable_check_crc32" = xno ; then
	AC_MSG_ERROR([For now, the CRC32 check must always be enabled.])
fi

m4_foreach([NAME], [SUPPORTED_CHECKS],
[AM_CONDITIONAL(COND_CHECK_[]m4_toupper(NAME), test "x$enable_check_[]NAME" = xyes)
])dnl

AC_MSG_CHECKING([if external SHA-256 should be used])
AC_ARG_ENABLE([external-sha256], AS_HELP_STRING([--enable-external-sha256],
		[Use SHA-256 code from the operating system.
		See INSTALL for possible subtle problems.]),
		[], [enable_external_sha256=no])
if test "x$enable_check_sha256" != "xyes"; then
	enable_external_sha256=no
fi
if test "x$enable_external_sha256" = xyes; then
	AC_MSG_RESULT([yes])
else
	AC_MSG_RESULT([no])
fi


#############
# MicroLZMA #
#############

AC_MSG_CHECKING([if MicroLZMA support should be built])
AC_ARG_ENABLE([microlzma], AS_HELP_STRING([--disable-microlzma],
		[Do not build MicroLZMA encoder and decoder.
		It is needed by specific applications only,
		for example, erofs-utils.]),
	[], [enable_microlzma=yes])
case $enable_microlzma in
	yes | no)
		AC_MSG_RESULT([$enable_microlzma])
		;;
	*)
		AC_MSG_RESULT([])
		AC_MSG_ERROR([--enable-microlzma accepts only `yes' or `no'.])
		;;
esac
AM_CONDITIONAL(COND_MICROLZMA, test "x$enable_microlzma" = xyes)


#############################
# .lz (lzip) format support #
#############################

AC_MSG_CHECKING([if .lz (lzip) decompression support should be built])
AC_ARG_ENABLE([lzip-decoder], AS_HELP_STRING([--disable-lzip-decoder],
		[Disable decompression support for .lz (lzip) files.]),
	[], [enable_lzip_decoder=yes])
if test "x$enable_decoder_lzma1" != xyes; then
	enable_lzip_decoder=no
	AC_MSG_RESULT([no because LZMA1 decoder is disabled])
elif test "x$enable_lzip_decoder" = xyes; then
	AC_DEFINE([HAVE_LZIP_DECODER], [1],
		[Define to 1 if .lz (lzip) decompression support is enabled.])
	AC_MSG_RESULT([yes])
else
	AC_MSG_RESULT([no])
fi
AM_CONDITIONAL(COND_LZIP_DECODER, test "x$enable_lzip_decoder" = xyes)


###########################
# Assembler optimizations #
###########################

AC_MSG_CHECKING([if assembler optimizations should be used])
AC_ARG_ENABLE([assembler], AS_HELP_STRING([--disable-assembler],
		[Do not use assembler optimizations even if such exist
		for the architecture.]),
	[], [enable_assembler=yes])
if test "x$enable_assembler" = xyes; then
	enable_assembler=no
	case $host_os in
		# Darwin should work too but only if not creating universal
		# binaries. Solaris x86 could work too but I cannot test.
		linux* | *bsd* | mingw* | cygwin | msys | *djgpp*)
			case $host_cpu in
				i?86)   enable_assembler=x86 ;;
			esac
			;;
	esac
fi
case $enable_assembler in
	x86 | no)
		AC_MSG_RESULT([$enable_assembler])
		;;
	*)
		AC_MSG_RESULT([])
		AC_MSG_ERROR([--enable-assembler accepts only `yes', `no', or `x86' (32-bit).])
		;;
esac
AM_CONDITIONAL(COND_ASM_X86, test "x$enable_assembler" = xx86)


#############
# CLMUL CRC #
#############

AC_ARG_ENABLE([clmul-crc], AS_HELP_STRING([--disable-clmul-crc],
		[Do not use carryless multiplication for CRC calculation
		even if support for it is detected.]),
	[], [enable_clmul_crc=yes])


#####################
# Size optimization #
#####################

AC_MSG_CHECKING([if small size is preferred over speed])
AC_ARG_ENABLE([small], AS_HELP_STRING([--enable-small],
		[Make liblzma smaller and a little slower.
		This is disabled by default to optimize for speed.]),
	[], [enable_small=no])
if test "x$enable_small" = xyes; then
	AC_DEFINE([HAVE_SMALL], [1], [Define to 1 if optimizing for size.])
elif test "x$enable_small" != xno; then
	AC_MSG_RESULT([])
	AC_MSG_ERROR([--enable-small accepts only `yes' or `no'])
fi
AC_MSG_RESULT([$enable_small])
AM_CONDITIONAL(COND_SMALL, test "x$enable_small" = xyes)


#############
# Threading #
#############

AC_MSG_CHECKING([if threading support is wanted])
AC_ARG_ENABLE([threads], AS_HELP_STRING([--enable-threads=METHOD],
		[Supported METHODS are `yes', `no', `posix', `win95', and
		`vista'. The default is `yes'. Using `no' together with
		--enable-small makes liblzma thread unsafe.]),
	[], [enable_threads=yes])

if test "x$enable_threads" = xyes; then
	case $host_os in
		mingw*)
			case $host_cpu in
				i?86)   enable_threads=win95 ;;
				*)      enable_threads=vista ;;
			esac
			;;
		*)
			enable_threads=posix
			;;
	esac
fi

case $enable_threads in
	posix | win95 | vista)
		AC_MSG_RESULT([yes, $enable_threads])
		;;
	no)
		AC_MSG_RESULT([no])
		;;
	*)
		AC_MSG_RESULT([])
		AC_MSG_ERROR([--enable-threads only accepts `yes', `no', `posix', `win95', or `vista'])
		;;
esac

# We use the actual result a little later.


#########################
# Assumed amount of RAM #
#########################

# We use 128 MiB as default, because it will allow decompressing files
# created with "xz -9". It would be slightly safer to guess a lower value,
# but most systems, on which we don't have any way to determine the amount
# of RAM, will probably have at least 128 MiB of RAM.
AC_MSG_CHECKING([how much RAM to assume if the real amount is unknown])
AC_ARG_ENABLE([assume-ram], AS_HELP_STRING([--enable-assume-ram=SIZE],
		[If and only if the real amount of RAM cannot be determined,
		assume SIZE MiB. The default is 128 MiB. This affects the
		default memory usage limit.]),
	[], [enable_assume_ram=128])
assume_ram_check=`echo "$enable_assume_ram" | tr -d 0123456789`
if test -z "$enable_assume_ram" || test -n "$assume_ram_check"; then
	AC_MSG_RESULT([])
	AC_MSG_ERROR([--enable-assume-ram accepts only an integer argument])
fi
AC_MSG_RESULT([$enable_assume_ram MiB])
AC_DEFINE_UNQUOTED([ASSUME_RAM], [$enable_assume_ram],
		[How many MiB of RAM to assume if the real amount cannot
		be determined.])


#########################
# Components to install #
#########################

AC_ARG_ENABLE([xz], [AS_HELP_STRING([--disable-xz],
		[do not build the xz tool])],
	[], [enable_xz=yes])
AM_CONDITIONAL([COND_XZ], [test x$enable_xz != xno])

AC_ARG_ENABLE([xzdec], [AS_HELP_STRING([--disable-xzdec],
		[do not build xzdec])],
	[], [enable_xzdec=yes])
test "x$enable_decoders" = xno && enable_xzdec=no
AM_CONDITIONAL([COND_XZDEC], [test x$enable_xzdec != xno])

AC_ARG_ENABLE([lzmadec], [AS_HELP_STRING([--disable-lzmadec],
		[do not build lzmadec
		(it exists primarily for LZMA Utils compatibility)])],
	[], [enable_lzmadec=yes])
test "x$enable_decoder_lzma1" = xno && enable_lzmadec=no
AM_CONDITIONAL([COND_LZMADEC], [test x$enable_lzmadec != xno])

AC_ARG_ENABLE([lzmainfo], [AS_HELP_STRING([--disable-lzmainfo],
		[do not build lzmainfo
		(it exists primarily for LZMA Utils compatibility)])],
	[], [enable_lzmainfo=yes])
test "x$enable_decoder_lzma1" = xno && enable_lzmainfo=no
AM_CONDITIONAL([COND_LZMAINFO], [test x$enable_lzmainfo != xno])

AC_ARG_ENABLE([lzma-links], [AS_HELP_STRING([--disable-lzma-links],
		[do not create symlinks for LZMA Utils compatibility])],
	[], [enable_lzma_links=yes])
AM_CONDITIONAL([COND_LZMALINKS], [test x$enable_lzma_links != xno])

AC_ARG_ENABLE([scripts], [AS_HELP_STRING([--disable-scripts],
		[do not install the scripts xzdiff, xzgrep, xzless, xzmore,
		and their symlinks])],
	[], [enable_scripts=yes])
AM_CONDITIONAL([COND_SCRIPTS], [test x$enable_scripts != xno])

AC_ARG_ENABLE([doc], [AS_HELP_STRING([--disable-doc],
		[do not install documentation files to docdir
		(man pages will still be installed)])],
	[], [enable_doc=yes])
AM_CONDITIONAL([COND_DOC], [test x$enable_doc != xno])


##############
# Sandboxing #
##############

AC_MSG_CHECKING([if sandboxing should be used])
AC_ARG_ENABLE([sandbox], [AS_HELP_STRING([--enable-sandbox=METHOD],
		[Sandboxing METHOD can be
		`auto', `no', `capsicum', or `pledge'.
		The default is `auto' which enables sandboxing if
		a supported sandboxing method is found.])],
	[], [enable_sandbox=auto])
case $enable_sandbox in
	auto)
		AC_MSG_RESULT([maybe (autodetect)])
		;;
	no | capsicum | pledge)
		AC_MSG_RESULT([$enable_sandbox])
		;;
	*)
		AC_MSG_RESULT([])
		AC_MSG_ERROR([--enable-sandbox only accepts `auto', `no', `capsicum', or `pledge'.])
		;;
esac


###########################
# PATH prefix for scripts #
###########################

# The scripts can add a prefix to the search PATH so that POSIX tools
# or the xz binary is always in the PATH.
AC_ARG_ENABLE([path-for-scripts],
	[AS_HELP_STRING([--enable-path-for-scripts=PREFIX],
		[If PREFIX isn't empty, PATH=PREFIX:$PATH will be set in
		the beginning of the scripts (xzgrep and others).
		The default is empty except on Solaris the default is
		/usr/xpg4/bin.])],
	[], [
		case $host_os in
			solaris*) enable_path_for_scripts=/usr/xpg4/bin ;;
			*)        enable_path_for_scripts= ;;
		esac
	])
if test -n "$enable_path_for_scripts" && test "x$enable_path_for_scripts" != xno ; then
	enable_path_for_scripts="PATH=$enable_path_for_scripts:\$PATH"
else
	enable_path_for_scripts=
fi
AC_SUBST([enable_path_for_scripts])


###############################################################################
# Checks for programs.
###############################################################################

echo
case $host_os in
	solaris*)
		# The gnulib POSIX shell macro below may pick a shell that
		# doesn't work with xzgrep. Workaround by picking a shell
		# that is known to work.
		if test -z "$gl_cv_posix_shell" && test -x /usr/xpg4/bin/sh; then
			gl_cv_posix_shell=/usr/xpg4/bin/sh
		fi
		;;
esac
gl_POSIX_SHELL
if test -z "$POSIX_SHELL" && test "x$enable_scripts" = xyes ; then
	AC_MSG_ERROR([No POSIX conforming shell (sh) was found.])
fi

echo
echo "Initializing Automake:"

# We don't use "subdir-objects" yet because it breaks "make distclean" when
# dependencies are enabled (as of Automake 1.14.1) due to this bug:
# https://debbugs.gnu.org/cgi/bugreport.cgi?bug=17354
# The -Wno-unsupported is used to silence warnings about missing
# "subdir-objects".
AM_INIT_AUTOMAKE([1.12 foreign tar-v7 filename-length-max=99 -Wno-unsupported])
AC_PROG_LN_S

dnl # Autoconf >= 2.70 warns that AC_PROG_CC_C99 is obsolete. However,
dnl # we have to keep using AC_PROG_CC_C99 instead of AC_PROG_CC
dnl # as long as we try to be compatible with Autoconf 2.69.
AC_PROG_CC_C99
if test x$ac_cv_prog_cc_c99 = xno ; then
	AC_MSG_ERROR([No C99 compiler was found.])
fi

AM_PROG_CC_C_O
AM_PROG_AS
AC_USE_SYSTEM_EXTENSIONS

AS_CASE([$enable_threads],
	[posix], [
		echo
		echo "POSIX threading support:"
		AX_PTHREAD([:]) dnl We don't need the HAVE_PTHREAD macro.
		LIBS="$LIBS $PTHREAD_LIBS"
		AM_CFLAGS="$AM_CFLAGS $PTHREAD_CFLAGS"

		dnl NOTE: PTHREAD_CC is ignored. It would be useful on AIX,
		dnl but it's tricky to get it right together with
		dnl AC_PROG_CC_C99. Thus, this is handled by telling the
		dnl user in INSTALL to set the correct CC manually.

		AC_DEFINE([MYTHREAD_POSIX], [1],
			[Define to 1 when using POSIX threads (pthreads).])

		# These are nice to have but not mandatory.
		#
		# FIXME: xz uses clock_gettime if it is available and can do
		# it even when threading is disabled. Moving this outside
		# of pthread detection may be undesirable because then
		# liblzma may get linked against librt even when librt isn't
		# needed by liblzma.
		OLD_CFLAGS=$CFLAGS
		CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
		AC_SEARCH_LIBS([clock_gettime], [rt])
		AC_CHECK_FUNCS([clock_gettime pthread_condattr_setclock])
		AC_CHECK_DECL([CLOCK_MONOTONIC], [AC_DEFINE(
			[HAVE_CLOCK_MONOTONIC], [1], [Define to 1 if
			`CLOCK_MONOTONIC' is declared in <time.h>.])], [],
			[[#include <time.h>]])
		CFLAGS=$OLD_CFLAGS
	],
	[win95], [
		AC_DEFINE([MYTHREAD_WIN95], [1], [Define to 1 when using
			Windows 95 (and thus XP) compatible threads.
			This avoids use of features that were added in
			Windows Vista.])
	],
	[vista], [
		AC_DEFINE([MYTHREAD_VISTA], [1], [Define to 1 when using
			Windows Vista compatible threads. This uses
			features that are not available on Windows XP.])
	]
)
AM_CONDITIONAL([COND_THREADS], [test "x$enable_threads" != xno])

echo
echo "Initializing Libtool:"
LT_PREREQ([2.4])
LT_INIT([win32-dll])
LT_LANG([Windows Resource])

# This is a bit wrong since it is possible to request that only some libs
# are built as shared. Using that feature isn't so common though, and this
# breaks only on Windows (at least for now) if the user enables only some
# libs as shared.
AM_CONDITIONAL([COND_SHARED], [test "x$enable_shared" != xno])

#####################
# Symbol versioning #
#####################

# NOTE: This checks if we are building shared or static library
# and if --with-pic or --without-pic was used. Thus this check
# must be after Libtool initialization.
AC_MSG_CHECKING([if library symbol versioning should be used])
AC_ARG_ENABLE([symbol-versions], [AS_HELP_STRING([--enable-symbol-versions],
		[Use symbol versioning for liblzma. Enabled by default on
		GNU/Linux, other GNU-based systems, and FreeBSD.])],
	[], [enable_symbol_versions=auto])
if test "x$enable_symbol_versions" = xauto; then
	case $host_os in
		# NOTE: Even if one omits -gnu on GNU/Linux (e.g.
		# i486-slackware-linux), configure will (via config.sub)
		# append -gnu (e.g. i486-slackware-linux-gnu), and this
		# test will work correctly.
		gnu* | *-gnu* | freebsd*)
			enable_symbol_versions=yes
			;;
		*)
			enable_symbol_versions=no
			;;
	esac
fi

# There are two variants for symbol versioning.
# See src/liblzma/validate_map.sh for details.
#
# On GNU/Linux, extra symbols are added in the C code. These extra symbols
# must not be put into a static library as they can cause problems (and
# even if they didn't cause problems, they would be useless). On other
# systems symbol versioning may be used too but there is no problem as only
# a linker script is specified in src/liblzma/Makefile.am and that isn't
# used when creating a static library.
#
# Libtool always uses -DPIC when building shared libraries by default and
# doesn't use it for static libs by default. This can be overridden with
# --with-pic and --without-pic though. As long as neither --with-pic nor
# --without-pic is used then we can use #ifdef PIC to detect if the file is
# being built for a shared library.
if test "x$enable_symbol_versions" = xno ; then
	enable_symbol_versions=no
	AC_MSG_RESULT([no])
elif test "x$enable_shared" = xno ; then
	enable_symbol_versions=no
	AC_MSG_RESULT([no (not building a shared library)])
else
	case "$host_cpu-$host_os" in
		microblaze*)
			# GCC 12 on MicroBlaze doesn't support __symver__
			# attribute. It's simplest and safest to use the
			# generic version on that platform since then only
			# the linker script is needed. The RHEL/CentOS 7
			# compatibility symbols don't matter on MicroBlaze.
			enable_symbol_versions=generic
			;;
		*-linux*)
			case "$pic_mode-$enable_static" in
				default-*)
					# Use symvers if PIC is defined.
					have_symbol_versions_linux=2
					;;
				*-no)
					# Not building static library.
					# Use symvers unconditionally.
					have_symbol_versions_linux=1
					;;
				*)
					AC_MSG_RESULT([])
					AC_MSG_ERROR([
    On GNU/Linux, building both shared and static library at the same time
    is not supported if --with-pic or --without-pic is used.
    Use either --disable-shared or --disable-static to build one type
    of library at a time. If both types are needed, build one at a time,
    possibly picking only src/liblzma/.libs/liblzma.a from the static build.])
					;;
			esac
			enable_symbol_versions=linux
			AC_DEFINE_UNQUOTED([HAVE_SYMBOL_VERSIONS_LINUX],
				[$have_symbol_versions_linux],
				[Define to 1 to if GNU/Linux-specific details
				are unconditionally wanted for symbol
				versioning. Define to 2 to if these are wanted
				only if also PIC is defined (allows building
				both shared and static liblzma at the same
				time with Libtool if neither --with-pic nor
				--without-pic is used). This define must be
				used together with liblzma_linux.map.])
			;;
		*)
			enable_symbol_versions=generic
			;;
	esac
	AC_MSG_RESULT([yes ($enable_symbol_versions)])
fi

AM_CONDITIONAL([COND_SYMVERS_LINUX],
	[test "x$enable_symbol_versions" = xlinux])
AM_CONDITIONAL([COND_SYMVERS_GENERIC],
	[test "x$enable_symbol_versions" = xgeneric])


###############################################################################
# Checks for libraries.
###############################################################################

dnl Support for _REQUIRE_VERSION was added in gettext 0.19.6. If both
dnl _REQUIRE_VERSION and _VERSION are present, the _VERSION is ignored.
dnl We use both for compatibility with other programs in the Autotools family.
echo
echo "Initializing gettext:"
AM_GNU_GETTEXT_REQUIRE_VERSION([0.19.6])
AM_GNU_GETTEXT_VERSION([0.19.6])
AM_GNU_GETTEXT([external])


###############################################################################
# Checks for header files.
###############################################################################

echo
echo "System headers and functions:"

# There is currently no workarounds in this package if some of
# these headers are missing.
AC_CHECK_HEADERS([fcntl.h limits.h sys/time.h],
	[],
	[AC_MSG_ERROR([Required header file(s) are missing.])])

# immintrin.h allows the use of the intrinsic functions if they are available.
# cpuid.h may be used for detecting x86 processor features at runtime.
AC_CHECK_HEADERS([immintrin.h cpuid.h])


###############################################################################
# Checks for typedefs, structures, and compiler characteristics.
###############################################################################

AC_HEADER_STDBOOL

AC_TYPE_UINT8_T
AC_TYPE_UINT16_T
AC_TYPE_INT32_T
AC_TYPE_UINT32_T
AC_TYPE_INT64_T
AC_TYPE_UINT64_T
AC_TYPE_UINTPTR_T

AC_CHECK_SIZEOF([size_t])

# The command line tool can copy high resolution timestamps if such
# information is available in struct stat. Otherwise one second accuracy
# is used.
AC_CHECK_MEMBERS([
	struct stat.st_atim.tv_nsec,
	struct stat.st_atimespec.tv_nsec,
	struct stat.st_atimensec,
	struct stat.st_uatime,
	struct stat.st_atim.st__tim.tv_nsec])

AC_SYS_LARGEFILE
AC_C_BIGENDIAN

# __attribute__((__constructor__)) can be used for one-time initializations.
# Use -Werror because some compilers accept unknown attributes and just
# give a warning.
#
# FIXME? Unfortunately -Werror can cause trouble if CFLAGS contains options
# that produce warnings for unrelated reasons. For example, GCC and Clang
# support -Wunused-macros which will warn about "#define _GNU_SOURCE 1"
# which will be among the #defines that Autoconf inserts to the beginning of
# the test program. There seems to be no nice way to prevent Autoconf from
# inserting the any defines to the test program.
AC_MSG_CHECKING([if __attribute__((__constructor__)) can be used])
have_func_attribute_constructor=no
OLD_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -Werror"
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
	__attribute__((__constructor__))
	static void my_constructor_func(void) { return; }
]])], [
	AC_DEFINE([HAVE_FUNC_ATTRIBUTE_CONSTRUCTOR], [1],
		[Define to 1 if __attribute__((__constructor__))
		is supported for functions.])
	have_func_attribute_constructor=yes
	AC_MSG_RESULT([yes])
], [
	AC_MSG_RESULT([no])
])
CFLAGS="$OLD_CFLAGS"


# The Win95 threading lacks a thread-safe one-time initialization function.
# The one-time initialization is needed for crc32_small.c and crc64_small.c
# create the CRC tables. So if small mode is enabled, the threading mode is
# win95, and the compiler does not support attribute constructor, then we
# would end up with a multithreaded build that is thread-unsafe. As a
# result this configuration is not allowed.
if test "x$enable_small$enable_threads$have_func_attribute_constructor" \
		= xyeswin95no; then
	AC_MSG_ERROR([
    --enable-threads=win95 and --enable-small cannot be used
    at the same time with a compiler that doesn't support
    __attribute__((__constructor__))])
fi

###############################################################################
# Checks for library functions.
###############################################################################

# Gnulib replacements as needed
gl_GETOPT

# Find the best function to set timestamps.
AC_CHECK_FUNCS([futimens futimes futimesat utimes _futime utime], [break])

# This is nice to have but not mandatory.
AC_CHECK_FUNCS([posix_fadvise])

TUKLIB_PROGNAME
TUKLIB_INTEGER
TUKLIB_PHYSMEM
TUKLIB_CPUCORES
TUKLIB_MBSTR

# If requested, check for system-provided SHA-256. At least the following
# implementations are supported:
#
# OS       Headers                     Library  Type           Function
# FreeBSD  sys/types.h + sha256.h      libmd    SHA256_CTX     SHA256_Init
# NetBSD   sys/types.h + sha2.h                 SHA256_CTX     SHA256_Init
# OpenBSD  sys/types.h + sha2.h                 SHA2_CTX       SHA256Init
# Solaris  sys/types.h + sha2.h        libmd    SHA256_CTX     SHA256Init
# MINIX 3  sys/types.h + sha2.h                 SHA256_CTX     SHA256_Init
# Darwin   CommonCrypto/CommonDigest.h          CC_SHA256_CTX  CC_SHA256_Init
#
# Note that Darwin's CC_SHA256_Update takes buffer size as uint32_t instead
# of size_t.
#
sha256_header_found=no
sha256_type_found=no
sha256_func_found=no
AS_IF([test "x$enable_external_sha256" = "xyes"], [
	# Test for Common Crypto before others, because Darwin has sha256.h
	# too and we don't want to use that, because on older versions it
	# uses OpenSSL functions, whose SHA256_Init is not guaranteed to
	# succeed.
	AC_CHECK_HEADERS(
		[CommonCrypto/CommonDigest.h sha256.h sha2.h],
		[sha256_header_found=yes ; break])
	if test "x$sha256_header_found" = xyes; then
		AC_CHECK_TYPES([CC_SHA256_CTX, SHA256_CTX, SHA2_CTX],
			[sha256_type_found=yes], [],
			[[#ifdef HAVE_SYS_TYPES_H
			  # include <sys/types.h>
			  #endif
			  #ifdef HAVE_COMMONCRYPTO_COMMONDIGEST_H
			  # include <CommonCrypto/CommonDigest.h>
			  #endif
			  #ifdef HAVE_SHA256_H
			  # include <sha256.h>
			  #endif
			  #ifdef HAVE_SHA2_H
			  # include <sha2.h>
			  #endif]])
		if test "x$sha256_type_found" = xyes ; then
			AC_SEARCH_LIBS([SHA256Init], [md])
			AC_SEARCH_LIBS([SHA256_Init], [md])
			AC_CHECK_FUNCS([CC_SHA256_Init SHA256Init SHA256_Init],
				[sha256_func_found=yes ; break])
		fi
	fi
])
AM_CONDITIONAL([COND_INTERNAL_SHA256], [test "x$sha256_func_found" = xno])
if test "x$enable_external_sha256$sha256_func_found" = xyesno; then
	AC_MSG_ERROR([--enable-external-sha256 was specified but no supported external SHA-256 implementation was found])
fi

# Check for SSE2 intrinsics. There is no run-time detection for SSE2 so if
# compiler options enable SSE2 then SSE2 support is required by the binaries.
# The compile-time check for SSE2 is done with #ifdefs because some compilers
# (ICC, MSVC) allow SSE2 intrinsics even when SSE2 isn't enabled.
AC_CHECK_DECL([_mm_movemask_epi8],
	[AC_DEFINE([HAVE__MM_MOVEMASK_EPI8], [1],
		[Define to 1 if _mm_movemask_epi8 is available.])],
	[],
[#ifdef HAVE_IMMINTRIN_H
#include <immintrin.h>
#endif])

# For faster CRC on 32/64-bit x86 and E2K (see also crc64_fast.c):
#
#   - Check for the CLMUL intrinsic _mm_clmulepi64_si128 in <immintrin.h>.
#     Check also for _mm_set_epi64x for consistency with CMake build
#     where it's needed to disable CLMUL with VS2013.
#
#   - Check that __attribute__((__target__("ssse3,sse4.1,pclmul"))) works
#     together with _mm_clmulepi64_si128 from <immintrin.h>. The attribute
#     was added in GCC 4.4 but some GCC 4.x versions don't allow intrinsics
#     with it. Exception: it must be not be used with EDG-based compilers
#     like ICC and the compiler on E2K.
#
# If everything above is supported, runtime detection will be used to keep the
# binaries working on systems that don't support the required extensions.
AC_MSG_CHECKING([if _mm_clmulepi64_si128 is usable])
AS_IF([test "x$enable_clmul_crc" = xno], [
	AC_MSG_RESULT([no, --disable-clmul-crc was used])
], [
	AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
#include <immintrin.h>

// CLMUL works on older E2K instruction set but it is slow due to emulation.
#if defined(__e2k__) && __iset__ < 6
#	error
#endif

// Intel's old compiler (ICC) can define __GNUC__ but the attribute must not
// be used with it. The new Clang-based ICX needs the attribute.
// Checking for !defined(__EDG__) catches ICC and other EDG-based compilers.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__EDG__)
__attribute__((__target__("ssse3,sse4.1,pclmul")))
#endif
__m128i my_clmul(__m128i a)
{
	const __m128i b = _mm_set_epi64x(1, 2);
	return _mm_clmulepi64_si128(a, b, 0);
}
	]])], [
		AC_DEFINE([HAVE_USABLE_CLMUL], [1],
			[Define to 1 if _mm_set_epi64x and
			_mm_clmulepi64_si128 are usable.
			See configure.ac for details.])
		AC_MSG_RESULT([yes])
	], [
		AC_MSG_RESULT([no])
	])
])

# Check for sandbox support. If one is found, set enable_sandbox=found.
AS_CASE([$enable_sandbox],
	[auto | capsicum], [
		AX_CHECK_CAPSICUM([enable_sandbox=found], [:])
	]
)
AS_CASE([$enable_sandbox],
	[auto | pledge], [
		AC_CHECK_FUNCS([pledge], [enable_sandbox=found ; break])
	]
)

# If a specific sandboxing method was explicitly requested and it wasn't
# found, give an error.
case $enable_sandbox in
	auto | no | found)
		;;
	*)
		AC_MSG_ERROR([$enable_sandbox support not found])
		;;
esac


###############################################################################
# If using GCC, set some additional AM_CFLAGS:
###############################################################################

if test "$GCC" = yes ; then
	echo
	echo "GCC extensions:"
fi

# Always do the visibility check but don't set AM_CFLAGS on Windows.
# This way things get set properly even on Windows.
gl_VISIBILITY
if test -n "$CFLAG_VISIBILITY" && test "$is_w32" = no; then
	AM_CFLAGS="$AM_CFLAGS $CFLAG_VISIBILITY"
fi

AS_IF([test "$GCC" = yes], [
	# Enable as much warnings as possible. These commented warnings won't
	# work for this package though:
	#   * -Wunreachable-code breaks several assert(0) cases, which are
	#     backed up with "return LZMA_PROG_ERROR".
	#   * -Wcast-qual would break various things where we need a non-const
	#     pointer although we don't modify anything through it.
	#   * -Winline, -Wdisabled-optimization, -Wunsafe-loop-optimizations
	#     don't seem so useful here; at least the last one gives some
	#     warnings which are not bugs.
	#   * -Wconversion still shows too many warnings.
	#
	# The flags before the empty line are for GCC and many of them
	# are supported by Clang too. The flags after the empty line are
	# for Clang.
	for NEW_FLAG in \
			-Wall \
			-Wextra \
			-Wvla \
			-Wc99-c11-compat \
			-Wformat=2 \
			-Winit-self \
			-Wmissing-include-dirs \
			-Wshift-overflow=2 \
			-Wstrict-overflow=3 \
			-Walloc-zero \
			-Wduplicated-cond \
			-Wfloat-equal \
			-Wundef \
			-Wshadow \
			-Wpointer-arith \
			-Wbad-function-cast \
			-Wwrite-strings \
			-Wdate-time \
			-Wsign-conversion \
			-Wfloat-conversion \
			-Wlogical-op \
			-Waggregate-return \
			-Wstrict-prototypes \
			-Wold-style-definition \
			-Wmissing-prototypes \
			-Wmissing-declarations \
			-Wredundant-decls \
			\
			-Wc99-compat \
			-Wc11-extensions \
			-Wc2x-compat \
			-Wc2x-extensions \
			-Wpre-c2x-compat \
			-Warray-bounds-pointer-arithmetic \
			-Wassign-enum \
			-Wconditional-uninitialized \
			-Wdocumentation \
			-Wduplicate-enum \
			-Wempty-translation-unit \
			-Wflexible-array-extensions \
			-Wmissing-variable-declarations \
			-Wnewline-eof \
			-Wshift-sign-overflow \
			-Wstring-conversion
	do
		AC_MSG_CHECKING([if $CC accepts $NEW_FLAG])
		OLD_CFLAGS="$CFLAGS"
		CFLAGS="$CFLAGS $NEW_FLAG -Werror"
		AC_COMPILE_IFELSE([AC_LANG_SOURCE(
				[[void foo(void); void foo(void) { }]])], [
			AM_CFLAGS="$AM_CFLAGS $NEW_FLAG"
			AC_MSG_RESULT([yes])
		], [
			AC_MSG_RESULT([no])
		])
		CFLAGS="$OLD_CFLAGS"
	done

	AC_ARG_ENABLE([werror],
		AS_HELP_STRING([--enable-werror], [Enable -Werror to abort
			compilation on all compiler warnings.]),
		[], [enable_werror=no])
	if test "x$enable_werror" = "xyes"; then
		AM_CFLAGS="$AM_CFLAGS -Werror"
	fi
])


###############################################################################
# Create the makefiles and config.h
###############################################################################

echo

# Don't build the lib directory at all if we don't need any replacement
# functions.
AM_CONDITIONAL([COND_GNULIB], test -n "$LIBOBJS")

# Add default AM_CFLAGS.
AC_SUBST([AM_CFLAGS])

# This is needed for src/scripts.
xz=`echo xz | sed "$program_transform_name"`
AC_SUBST([xz])

AC_CONFIG_FILES([
	Makefile
	po/Makefile.in
	lib/Makefile
	src/Makefile
	src/liblzma/Makefile
	src/liblzma/api/Makefile
	src/xz/Makefile
	src/xzdec/Makefile
	src/lzmainfo/Makefile
	src/scripts/Makefile
	tests/Makefile
	debug/Makefile
])
AC_CONFIG_FILES([src/scripts/xzdiff], [chmod +x src/scripts/xzdiff])
AC_CONFIG_FILES([src/scripts/xzgrep], [chmod +x src/scripts/xzgrep])
AC_CONFIG_FILES([src/scripts/xzmore], [chmod +x src/scripts/xzmore])
AC_CONFIG_FILES([src/scripts/xzless], [chmod +x src/scripts/xzless])

AC_OUTPUT

# Some warnings
if test x$tuklib_cv_physmem_method = xunknown; then
	echo
	echo "WARNING:"
	echo "No supported method to detect the amount of RAM."
	echo "Consider using --enable-assume-ram (if you didn't already)"
	echo "or make a patch to add support for this operating system."
fi

if test x$tuklib_cv_cpucores_method = xunknown; then
	echo
	echo "WARNING:"
	echo "No supported method to detect the number of CPU cores."
fi

if test "x$enable_threads$enable_small$have_func_attribute_constructor" \
		= xnoyesno; then
	echo
	echo "NOTE:"
	echo "liblzma will be thread-unsafe due to the combination"
	echo "of --disable-threads --enable-small when using a compiler"
	echo "that doesn't support __attribute__((__constructor__))."
fi
