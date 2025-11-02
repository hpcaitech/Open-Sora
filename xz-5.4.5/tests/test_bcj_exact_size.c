///////////////////////////////////////////////////////////////////////////////
//
/// \file       test_bcj_exact_size.c
/// \brief      Tests BCJ decoding when the output size is known
///
/// These tests fail with XZ Utils 5.0.3 and earlier.
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "tests.h"


static void
test_exact_size(void)
{
#if !defined(HAVE_ENCODERS) || !defined(HAVE_DECODERS)
	assert_skip("Encoder or decoder support disabled");
#else
	if (!lzma_filter_encoder_is_supported(LZMA_FILTER_POWERPC)
			|| !lzma_filter_decoder_is_supported(
				LZMA_FILTER_POWERPC))
		assert_skip("PowerPC BCJ encoder and/or decoder "
				"is disabled");

	// Something to be compressed
	const uint8_t in[16] = "0123456789ABCDEF";

	// in[] after compression
	uint8_t compressed[1024];
	size_t compressed_size = 0;

	// Output buffer for decompressing compressed[]
	uint8_t out[sizeof(in)];

	// Compress with PowerPC BCJ and LZMA2. PowerPC BCJ is used because
	// it has fixed 4-byte alignment which makes triggering the potential
	// bug easy.
	lzma_options_lzma opt_lzma2;
	assert_false(lzma_lzma_preset(&opt_lzma2, 0));

	lzma_filter filters[3] = {
		{ .id = LZMA_FILTER_POWERPC, .options = NULL },
		{ .id = LZMA_FILTER_LZMA2, .options = &opt_lzma2 },
		{ .id = LZMA_VLI_UNKNOWN, .options = NULL },
	};

	assert_lzma_ret(lzma_stream_buffer_encode(
			filters, LZMA_CHECK_CRC32, NULL,
			in, sizeof(in),
			compressed, &compressed_size, sizeof(compressed)),
		LZMA_OK);

	// Decompress so that we won't give more output space than
	// the Stream will need.
	lzma_stream strm = LZMA_STREAM_INIT;
	assert_lzma_ret(lzma_stream_decoder(&strm, 10 << 20, 0), LZMA_OK);

	strm.next_in = compressed;
	strm.next_out = out;

	while (true) {
		if (strm.total_in < compressed_size)
			strm.avail_in = 1;

		const lzma_ret ret = lzma_code(&strm, LZMA_RUN);
		if (ret == LZMA_STREAM_END) {
			assert_uint_eq(strm.total_in, compressed_size);
			assert_uint_eq(strm.total_out, sizeof(in));
			lzma_end(&strm);
			return;
		}

		assert_lzma_ret(ret, LZMA_OK);

		if (strm.total_out < sizeof(in))
			strm.avail_out = 1;
	}
#endif
}


static void
test_empty_block(void)
{
#ifndef HAVE_DECODERS
	assert_skip("Decoder support disabled");
#else
	if (!lzma_filter_decoder_is_supported(LZMA_FILTER_POWERPC))
		assert_skip("PowerPC BCJ decoder is disabled");

	// An empty file with one Block using PowerPC BCJ and LZMA2.
	size_t in_size;
	uint8_t *empty_bcj_lzma2 = tuktest_file_from_srcdir(
			"files/good-1-empty-bcj-lzma2.xz", &in_size);

	// Decompress without giving any output space.
	uint64_t memlimit = 1 << 20;
	uint8_t out[1];
	size_t in_pos = 0;
	size_t out_pos = 0;
	assert_lzma_ret(lzma_stream_buffer_decode(&memlimit, 0, NULL,
			empty_bcj_lzma2, &in_pos, in_size, out, &out_pos, 0),
		LZMA_OK);
	assert_uint_eq(in_pos, in_size);
	assert_uint_eq(out_pos, 0);
#endif
}


extern int
main(int argc, char **argv)
{
	tuktest_start(argc, argv);

	tuktest_run(test_exact_size);
	tuktest_run(test_empty_block);

	return tuktest_end();
}
