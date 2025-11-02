#!/bin/sh

###############################################################################
#
# Author: Lasse Collin
#
# This file has been put into the public domain.
# You can do whatever you want with this file.
#
###############################################################################

# If xz wasn't built, this test is skipped.
if test -x ../src/xz/xz ; then
	:
else
	exit 77
fi

# If compression or decompression support is missing, this test is skipped.
# This isn't perfect as if only some compressors or decompressors are disabled
# then this script can still fail because for now this doesn't check the
# availability of each filter.
if grep 'define HAVE_ENCODERS' ../config.h > /dev/null \
		&& grep 'define HAVE_DECODERS' ../config.h > /dev/null ; then
	:
else
	echo "Compression or decompression support is disabled, skipping this test."
	exit 77
fi

# Find out if our shell supports functions.
eval 'unset foo ; foo() { return 42; } ; foo'
if test $? != 42 ; then
	echo "/bin/sh doesn't support functions, skipping this test."
	exit 77
fi

test_xz() {
	if $XZ -c "$@" "$FILE" > "$TMP_COMP"; then
		:
	else
		echo "Compressing failed: $* $FILE"
		exit 1
	fi

	if $XZ -cd "$TMP_COMP" > "$TMP_UNCOMP" ; then
		:
	else
		echo "Decompressing failed: $* $FILE"
		exit 1
	fi

	if cmp "$TMP_UNCOMP" "$FILE" ; then
		:
	else
		echo "Decompressed file does not match" \
				"the original: $* $FILE"
		exit 1
	fi

	if test -n "$XZDEC" ; then
		if $XZDEC "$TMP_COMP" > "$TMP_UNCOMP" ; then
			:
		else
			echo "Decompressing failed: $* $FILE"
			exit 1
		fi

		if cmp "$TMP_UNCOMP" "$FILE" ; then
			:
		else
			echo "Decompressed file does not match" \
					"the original: $* $FILE"
			exit 1
		fi
	fi
}

XZ="../src/xz/xz --memlimit-compress=48MiB --memlimit-decompress=5MiB \
		--no-adjust --threads=1 --check=crc32"
grep "define HAVE_CHECK_CRC64" ../config.h > /dev/null \
		&& XZ="$XZ --check=crc64"
XZDEC="../src/xzdec/xzdec" # No memory usage limiter available
test -x ../src/xzdec/xzdec || XZDEC=

# Create the required input file if needed.
#
# Derive temporary filenames for compressed and uncompressed outputs
# from the input filename. This is needed when multiple tests are
# run in parallel.
FILE=$1
TMP_COMP="tmp_comp_$FILE"
TMP_UNCOMP="tmp_uncomp_$FILE"

case $FILE in
	# compress_generated files will be created in the build directory
	# in the /tests/ sub-directory.
	compress_generated_*)
		if ./create_compress_files "$FILE" ; then
			:
		else
			rm -f "$FILE"
			echo "Failed to create the file '$FILE'."
			exit 1
		fi
		;;
	# compress_prepared files exist in the source directory since they
	# do not need to be copied or regenerated.
	compress_prepared_*)
		FILE="$srcdir/$FILE"
		;;
	'')
		echo "No test file was specified."
		exit 1
		;;
esac

# Remove temporary now (in case they are something weird), and on exit.
rm -f "$TMP_COMP" "$TMP_UNCOMP"
trap 'rm -f "$TMP_COMP" "$TMP_UNCOMP"' 0

# Compress and decompress the file with various filter configurations.
#
# Don't test with empty arguments; it breaks some ancient
# proprietary /bin/sh versions due to $@ used in test_xz().
test_xz -1
test_xz -2
test_xz -3
test_xz -4

test_filter()
{
	grep "define HAVE_ENCODER_$1 1" ../config.h > /dev/null || return
	grep "define HAVE_DECODER_$1 1" ../config.h > /dev/null || return
	shift
	test_xz "$@" --lzma2=dict=64KiB,nice=32,mode=fast
}

test_filter DELTA --delta=dist=1
test_filter DELTA --delta=dist=4
test_filter DELTA --delta=dist=256
test_filter X86 --x86
test_filter POWERPC --power
test_filter IA64 --ia64
test_filter ARM --arm
test_filter ARMTHUMB --armthumb
test_filter ARM64 --arm64
test_filter SPARC --sparc

exit 0
