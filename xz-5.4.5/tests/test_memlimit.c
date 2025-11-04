///////////////////////////////////////////////////////////////////////////////
//
/// \file       test_memlimit.c
/// \brief      Tests memory usage limit in decoders
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "tests.h"
#include "mythread.h"


#define MEMLIMIT_TOO_LOW 1234U
#define MEMLIMIT_HIGH_ENOUGH (2U << 20)


static uint8_t *in;
static size_t in_size;

#ifdef HAVE_DECODERS
static uint8_t out[8192];
#endif


static void
test_memlimit_stream_decoder(void)
{
#ifndef HAVE_DECODERS
	assert_skip("Decoder support disabled");
#else
	lzma_stream strm = LZMA_STREAM_INIT;
	assert_lzma_ret(lzma_stream_decoder(&strm, MEMLIMIT_TOO_LOW, 0),
			LZMA_OK);

	strm.next_in = in;
	strm.avail_in = in_size;
	strm.next_out = out;
	strm.avail_out = sizeof(out);

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_MEMLIMIT_ERROR);

	assert_uint_eq(lzma_memlimit_get(&strm), MEMLIMIT_TOO_LOW);
	assert_lzma_ret(lzma_memlimit_set(&strm, MEMLIMIT_TOO_LOW + 1),
			LZMA_MEMLIMIT_ERROR);
	assert_lzma_ret(lzma_memlimit_set(&strm, MEMLIMIT_HIGH_ENOUGH),
			LZMA_OK);

	// This fails before commit 660739f99ab211edec4071de98889fb32ed04e98
	// (liblzma <= 5.2.6, liblzma <= 5.3.3alpha). It was fixed in 5.2.7.
	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);

	lzma_end(&strm);
#endif
}


static void
test_memlimit_stream_decoder_mt(void)
{
#ifndef MYTHREAD_ENABLED
	assert_skip("Threading support disabled");
#elif !defined(HAVE_DECODERS)
	assert_skip("Decoder support disabled");
#else
	lzma_stream strm = LZMA_STREAM_INIT;
	lzma_mt mt = {
		.flags = 0,
		.threads = 1,
		.timeout = 0,
		.memlimit_threading = 0,
		.memlimit_stop = MEMLIMIT_TOO_LOW,
	};

	assert_lzma_ret(lzma_stream_decoder_mt(&strm, &mt), LZMA_OK);

	strm.next_in = in;
	strm.avail_in = in_size;
	strm.next_out = out;
	strm.avail_out = sizeof(out);

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_MEMLIMIT_ERROR);

	assert_uint_eq(lzma_memlimit_get(&strm), MEMLIMIT_TOO_LOW);
	assert_lzma_ret(lzma_memlimit_set(&strm, MEMLIMIT_TOO_LOW + 1),
			LZMA_MEMLIMIT_ERROR);
	assert_lzma_ret(lzma_memlimit_set(&strm, MEMLIMIT_HIGH_ENOUGH),
			LZMA_OK);

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);
	lzma_end(&strm);
#endif
}


static void
test_memlimit_alone_decoder(void)
{
#ifndef HAVE_DECODERS
	assert_skip("Decoder support disabled");
#else
	size_t alone_size;
	uint8_t *alone_buf = tuktest_file_from_srcdir(
			"files/good-unknown_size-with_eopm.lzma", &alone_size);

	lzma_stream strm = LZMA_STREAM_INIT;
	assert_lzma_ret(lzma_alone_decoder(&strm, MEMLIMIT_TOO_LOW), LZMA_OK);

	strm.next_in = alone_buf;
	strm.avail_in = alone_size;
	strm.next_out = out;
	strm.avail_out = sizeof(out);

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_MEMLIMIT_ERROR);

	assert_uint_eq(lzma_memlimit_get(&strm), MEMLIMIT_TOO_LOW);
	assert_lzma_ret(lzma_memlimit_set(&strm, MEMLIMIT_TOO_LOW + 1),
			LZMA_MEMLIMIT_ERROR);
	assert_lzma_ret(lzma_memlimit_set(&strm, MEMLIMIT_HIGH_ENOUGH),
			LZMA_OK);

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);
	lzma_end(&strm);
#endif
}


static void
test_memlimit_auto_decoder(void)
{
#ifndef HAVE_DECODERS
	assert_skip("Decoder support disabled");
#else
	lzma_stream strm = LZMA_STREAM_INIT;
	assert_lzma_ret(lzma_auto_decoder(&strm, MEMLIMIT_TOO_LOW, 0),
			LZMA_OK);

	strm.next_in = in;
	strm.avail_in = in_size;
	strm.next_out = out;
	strm.avail_out = sizeof(out);

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_MEMLIMIT_ERROR);

	assert_uint_eq(lzma_memlimit_get(&strm), MEMLIMIT_TOO_LOW);
	assert_lzma_ret(lzma_memlimit_set(&strm, MEMLIMIT_TOO_LOW + 1),
			LZMA_MEMLIMIT_ERROR);
	assert_lzma_ret(lzma_memlimit_set(&strm, MEMLIMIT_HIGH_ENOUGH),
			LZMA_OK);

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);
	lzma_end(&strm);
#endif
}


extern int
main(int argc, char **argv)
{
	tuktest_start(argc, argv);

	in = tuktest_file_from_srcdir("files/good-1-check-crc32.xz", &in_size);

	tuktest_run(test_memlimit_stream_decoder);
	tuktest_run(test_memlimit_stream_decoder_mt);
	tuktest_run(test_memlimit_alone_decoder);
	tuktest_run(test_memlimit_auto_decoder);

	return tuktest_end();
}
