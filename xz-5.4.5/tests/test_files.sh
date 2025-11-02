#!/bin/sh

###############################################################################
#
# Author: Lasse Collin
#
# This file has been put into the public domain.
# You can do whatever you want with this file.
#
###############################################################################

# If both xz and xzdec were not build, skip this test.
XZ=../src/xz/xz
XZDEC=../src/xzdec/xzdec
test -x "$XZ" || XZ=
test -x "$XZDEC" || XZDEC=
if test -z "$XZ$XZDEC"; then
	echo "xz and xzdec were not built, skipping this test."
	exit 77
fi

# If decompression support is missing, this test is skipped.
# This isn't perfect as if only some decompressors are disabled
# then some good files might not decompress and the test fails
# for a (kind of) wrong reason.
if grep 'define HAVE_DECODERS' ../config.h > /dev/null ; then
	:
else
	echo "Decompression support is disabled, skipping this test."
	exit 77
fi

# If a feature was disabled at build time, make it possible to skip
# some of the test files. Use exit status 77 if any files were skipped.
EXIT_STATUS=0
have_feature()
{
	grep "define HAVE_$1 1" ../config.h > /dev/null && return 0
	printf '%s: Skipping because HAVE_%s is not enabled\n' "$2" "$1"
	EXIT_STATUS=77
	return 1
}


#######
# .xz #
#######

# If these integrity check types were disabled at build time,
# allow the tests to pass still.
NO_WARN=
grep 'define HAVE_CHECK_CRC64' ../config.h > /dev/null || NO_WARN=-qQ
grep 'define HAVE_CHECK_SHA256' ../config.h > /dev/null || NO_WARN=-qQ

for I in "$srcdir"/files/good-*.xz
do
	# If features were disabled at build time, keep this still working.
	case $I in
		*/good-1-*delta-lzma2*.xz)
			have_feature DECODER_DELTA "$I" || continue
			;;
	esac
	case $I in
		*/good-1-empty-bcj-lzma2.xz)
			have_feature DECODER_POWERPC "$I" || continue
			;;
	esac
	case $I in
		*/good-1-sparc-lzma2.xz)
			have_feature DECODER_SPARC "$I" || continue
			;;
	esac
	case $I in
		*/good-1-x86-lzma2.xz)
			have_feature DECODER_X86 "$I" || continue
			;;
	esac
	case $I in
		*/good-1-arm64-lzma2-*.xz)
			have_feature DECODER_ARM64 "$I" || continue
			;;
	esac

	if test -z "$XZ" || "$XZ" $NO_WARN -dc "$I" > /dev/null; then
		:
	else
		echo "Good file failed: $I"
		exit 1
	fi

	if test -z "$XZDEC" || "$XZDEC" $NO_WARN "$I" > /dev/null; then
		:
	else
		echo "Good file failed: $I"
		exit 1
	fi
done

for I in "$srcdir"/files/bad-*.xz
do
	if test -n "$XZ" && "$XZ" -dc "$I" > /dev/null 2>&1; then
		echo "Bad file succeeded: $I"
		exit 1
	fi

	# xzdec doesn't warn about unsupported check so skip this if any of
	# the check types were disabled at built time (NO_WARN isn't empty).
	if test -n "$XZDEC" && test -z "$NO_WARN" \
			&& "$XZDEC" "$I" > /dev/null 2>&1; then
		echo "Bad file succeeded: $I"
		exit 1
	fi
done

# Testing for the lzma_index_append() bug in <= 5.2.6 needs "xz -l":
I="$srcdir/files/bad-3-index-uncomp-overflow.xz"
if test -n "$XZ" && "$XZ" -l "$I" > /dev/null 2>&1; then
	echo "Bad file succeeded with xz -l: $I"
	exit 1
fi

for I in "$srcdir"/files/unsupported-*.xz
do
	# Test these only with xz as unsupported-check.xz will exit
	# successfully with xzdec because it doesn't warn about
	# unsupported check type.
	if test -n "$XZ" && "$XZ" -dc "$I" > /dev/null 2>&1; then
		echo "Unsupported file succeeded: $I"
		exit 1
	fi
done

# Test that this passes with --no-warn (-Q).
I="$srcdir/files/unsupported-check.xz"
if test -z "$XZ" || "$XZ" -dcqQ "$I" > /dev/null; then
	:
else
	echo "Unsupported file failed with xz -Q: $I"
	exit 1
fi

if test -z "$XZDEC" || "$XZDEC" -qQ "$I" > /dev/null; then
	:
else
	echo "Unsupported file failed with xzdec -Q: $I"
	exit 1
fi


#########
# .lzma #
#########

for I in "$srcdir"/files/good-*.lzma
do
	if test -z "$XZ" || "$XZ" -dc "$I" > /dev/null; then
		:
	else
		echo "Good file failed: $I"
		exit 1
	fi
done

for I in "$srcdir"/files/bad-*.lzma
do
	if test -n "$XZ" && "$XZ" -dc "$I" > /dev/null 2>&1; then
		echo "Bad file succeeded: $I"
		exit 1
	fi
done


#######
# .lz #
#######

if have_feature LZIP_DECODER ".lz files" ; then
	for I in "$srcdir"/files/good-*.lz
	do
		if test -z "$XZ" || "$XZ" -dc "$I" > /dev/null; then
			:
		else
			echo "Good file failed: $I"
			exit 1
		fi
	done

	for I in "$srcdir"/files/bad-*.lz "$srcdir"/files/unsupported-*.lz
	do
		if test -n "$XZ" && "$XZ" -dc "$I" > /dev/null 2>&1; then
			echo "Bad file succeeded: $I"
			exit 1
		fi
	done
fi

exit "$EXIT_STATUS"
