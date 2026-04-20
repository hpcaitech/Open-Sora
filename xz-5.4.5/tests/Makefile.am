##
## Author: Lasse Collin
##
## This file has been put into the public domain.
## You can do whatever you want with this file.
##

EXTRA_DIST = \
	files \
	ossfuzz \
	tuktest.h \
	tests.h \
	test_files.sh \
	test_compress.sh \
	test_compress_prepared_bcj_sparc \
	test_compress_prepared_bcj_x86 \
	test_compress_generated_abc \
	test_compress_generated_random \
	test_compress_generated_text \
	test_scripts.sh \
	bcj_test.c \
	compress_prepared_bcj_sparc \
	compress_prepared_bcj_x86 \
	xzgrep_expected_output

AM_CPPFLAGS = \
	-I$(top_srcdir)/src/common \
	-I$(top_srcdir)/src/liblzma/api \
	-I$(top_srcdir)/src/liblzma

LDADD = $(top_builddir)/src/liblzma/liblzma.la

LDADD += $(LTLIBINTL)

check_PROGRAMS = \
	create_compress_files \
	test_check \
	test_hardware \
	test_stream_flags \
	test_filter_flags \
	test_filter_str \
	test_block_header \
	test_index \
	test_index_hash \
	test_bcj_exact_size \
	test_memlimit \
	test_lzip_decoder \
	test_vli

TESTS = \
	test_check \
	test_hardware \
	test_stream_flags \
	test_filter_flags \
	test_filter_str \
	test_block_header \
	test_index \
	test_index_hash \
	test_bcj_exact_size \
	test_memlimit \
	test_lzip_decoder \
	test_vli \
	test_files.sh \
	test_compress_prepared_bcj_sparc \
	test_compress_prepared_bcj_x86 \
	test_compress_generated_abc \
	test_compress_generated_random \
	test_compress_generated_text

if COND_SCRIPTS
TESTS += test_scripts.sh
endif

clean-local:
	-rm -f compress_generated_* \
		xzgrep_test_output xzgrep_test_1.xz xzgrep_test_2.xz
