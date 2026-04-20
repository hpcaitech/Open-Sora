#!/bin/sh

###############################################################################
#
# Check liblzma_*.map for certain types of errors.
#
# liblzma_generic.map is for FreeBSD and Solaris and possibly others
# except GNU/Linux.
#
# liblzma_linux.map is for GNU/Linux only. This and the matching extra code
# in the .c files make liblzma >= 5.2.7 compatible with binaries that were
# linked against ill-patched liblzma in RHEL/CentOS 7. By providing the
# compatibility in official XZ Utils release will hopefully prevent people
# from further copying the broken patch to other places when they want
# compatibility with binaries linked on RHEL/CentOS 7. The long version
# of the story:
#
#     RHEL/CentOS 7 shipped with 5.1.2alpha, including the threaded
#     encoder that is behind #ifdef LZMA_UNSTABLE in the API headers.
#     In 5.1.2alpha these symbols are under XZ_5.1.2alpha in liblzma.map.
#     API/ABI compatibility tracking isn't done between development
#     releases so newer releases didn't have XZ_5.1.2alpha anymore.
#
#     Later RHEL/CentOS 7 updated xz to 5.2.2 but they wanted to keep
#     the exported symbols compatible with 5.1.2alpha. After checking
#     the ABI changes it turned out that >= 5.2.0 ABI is backward
#     compatible with the threaded encoder functions from 5.1.2alpha
#     (but not vice versa as fixes and extensions to these functions
#     were made between 5.1.2alpha and 5.2.0).
#
#     In RHEL/CentOS 7, XZ Utils 5.2.2 was patched with
#     xz-5.2.2-compat-libs.patch to modify liblzma.map:
#
#       - XZ_5.1.2alpha was added with lzma_stream_encoder_mt and
#         lzma_stream_encoder_mt_memusage. This matched XZ Utils 5.1.2alpha.
#
#       - XZ_5.2 was replaced with XZ_5.2.2. It is clear that this was
#         an error; the intention was to keep using XZ_5.2 (XZ_5.2.2
#         has never been used in XZ Utils). So XZ_5.2.2 lists all
#         symbols that were listed under XZ_5.2 before the patch.
#         lzma_stream_encoder_mt and _mt_memusage are included too so
#         they are listed both here and under XZ_5.1.2alpha.
#
#     The patch didn't add any __asm__(".symver ...") lines to the .c
#     files. Thus the resulting liblzma.so exports the threaded encoder
#     functions under XZ_5.1.2alpha only. Listing the two functions
#     also under XZ_5.2.2 in liblzma.map has no effect without
#     matching .symver lines.
#
#     The lack of XZ_5.2 in RHEL/CentOS 7 means that binaries linked
#     against unpatched XZ Utils 5.2.x won't run on RHEL/CentOS 7.
#     This is unfortunate but this alone isn't too bad as the problem
#     is contained within RHEL/CentOS 7 and doesn't affect users
#     of other distributions. It could also be fixed internally in
#     RHEL/CentOS 7.
#
#     The second problem is more serious: In XZ Utils 5.2.2 the API
#     headers don't have #ifdef LZMA_UNSTABLE for obvious reasons.
#     This is true in RHEL/CentOS 7 version too. Thus now programs
#     using new APIs can be compiled without an extra #define. However,
#     the programs end up depending on symbol version XZ_5.1.2alpha
#     (and possibly also XZ_5.2.2) instead of XZ_5.2 as they would
#     with an unpatched XZ Utils 5.2.2. This means that such binaries
#     won't run on other distributions shipping XZ Utils >= 5.2.0 as
#     they don't provide XZ_5.1.2alpha or XZ_5.2.2; they only provide
#     XZ_5.2 (and XZ_5.0). (This includes RHEL/CentOS 8 as the patch
#     luckily isn't included there anymore with XZ Utils 5.2.4.)
#
#     Binaries built by RHEL/CentOS 7 users get distributed and then
#     people wonder why they don't run on some other distribution.
#     Seems that people have found out about the patch and been copying
#     it to some build scripts, seemingly curing the symptoms but
#     actually spreading the illness further and outside RHEL/CentOS 7.
#     Adding compatibility in an official XZ Utils release should work
#     as a vaccine against this ill patch and stop it from spreading.
#     The vaccine is kept GNU/Linux-only as other OSes should be immune
#     (hopefully it hasn't spread via some build script to other OSes).
#
# Author: Lasse Collin
#
# This file has been put into the public domain.
# You can do whatever you want with this file.
#
###############################################################################

LC_ALL=C
export LC_ALL

STATUS=0

cd "$(dirname "$0")"

# Get the list of symbols that aren't defined in liblzma_generic.map.
SYMS=$(sed -n 's/^extern LZMA_API([^)]*) \([a-z0-9_]*\)(.*$/\1;/p' \
		api/lzma/*.h \
	| sort \
	| grep -Fve "$(sed '/[{}:*]/d;/^$/d;s/^	//' liblzma_generic.map)")

# Check that there are no old alpha or beta versions listed.
VER=$(cd ../.. && sh build-aux/version.sh)
NAMES=
case $VER in
	*alpha | *beta)
		NAMES=$(sed -n 's/^.*XZ_\([^ ]*\)\(alpha\|beta\) .*$/\1\2/p' \
			liblzma_generic.map | grep -Fv "$VER")
		;;
esac

# Check for duplicate lines. It can catch missing dependencies.
DUPS=$(sort liblzma_generic.map | sed '/^$/d;/^global:$/d' | uniq -d)

# Check that liblzma_linux.map is in sync with liblzma_generic.map.
# The RHEL/CentOS 7 compatibility symbols are in a fixed location
# so it makes it easy to remove them for comparison with liblzma_generic.map.
#
# NOTE: Putting XZ_5.2 before the compatibility symbols XZ_5.1.2alpha
# and XZ_5.2.2 in liblzma_linux.map is important: If liblzma_linux.map is
# incorrectly used without #define HAVE_SYMBOL_VERSIONS_LINUX, only the first
# occurrence of each function name will be used from liblzma_linux.map;
# the rest are ignored by the linker. Thus having XZ_5.2 before the
# compatibility symbols means that @@XZ_5.2 will be used for the symbols
# listed under XZ_5.2 {...} and the same function names later in
# the file under XZ_5.1.2alpha {...} and XZ_5.2.2 {...} will be
# ignored (@XZ_5.1.2alpha or @XZ_5.2.2 won't be added at all when
# the #define HAVE_SYMBOL_VERSIONS_LINUX isn't used).
IN_SYNC=
if ! sed '109,123d' liblzma_linux.map \
		| cmp -s - liblzma_generic.map; then
	IN_SYNC=no
fi

# Print error messages if needed.
if test -n "$SYMS$NAMES$DUPS$IN_SYNC"; then
	echo
	echo 'validate_map.sh found problems from liblzma_*.map:'
	echo

	if test -n "$SYMS"; then
		echo 'liblzma_generic.map lacks the following symbols:'
		echo "$SYMS"
		echo
	fi

	if test -n "$NAMES"; then
		echo 'Obsolete alpha or beta version names:'
		echo "$NAMES"
		echo
	fi

	if test -n "$DUPS"; then
		echo 'Duplicate lines:'
		echo "$DUPS"
		echo
	fi

	if test -n "$IN_SYNC"; then
		echo "liblzma_generic.map and liblzma_linux.map aren't in sync"
		echo
	fi

	STATUS=1
fi

# Exit status is 1 if problems were found, 0 otherwise.
exit "$STATUS"
