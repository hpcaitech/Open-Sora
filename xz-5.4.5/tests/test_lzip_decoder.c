///////////////////////////////////////////////////////////////////////////////
//
/// \file       test_lzip_decoder.c
/// \brief      Tests decoding lzip data
//
//  Author:     Jia Tan
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "tests.h"

#ifdef HAVE_LZIP_DECODER

// Memlimit large enough to pass all of the test files
#define MEMLIMIT (1U << 20)
#define DECODE_CHUNK_SIZE 1024


// The uncompressed data in the test files are short US-ASCII strings.
// The tests check if the decompressed output is what it is expected to be.
// Storing the strings here as text would break the tests on EBCDIC systems
// and storing the strings as an array of hex values is inconvenient, so
// store the CRC32 values of the expected data instead.
//
// CRC32 value of "Hello\nWorld\n"
static const uint32_t hello_world_crc = 0x15A2A343;

// CRC32 value of "Trailing garbage\n"
static const uint32_t trailing_garbage_crc = 0x87081A60;


// Helper function to decode a good file with no flags and plenty high memlimit
static void
basic_lzip_decode(const char *src, const uint32_t expected_crc)
{
	size_t file_size;
	uint8_t *data = tuktest_file_from_srcdir(src, &file_size);
	uint32_t checksum = 0;

	lzma_stream strm = LZMA_STREAM_INIT;
	assert_lzma_ret(lzma_lzip_decoder(&strm, MEMLIMIT, 0), LZMA_OK);

	uint8_t *output_buffer = tuktest_malloc(DECODE_CHUNK_SIZE);

	strm.next_in = data;
	strm.next_out = output_buffer;
	strm.avail_out = DECODE_CHUNK_SIZE;

	// Feed 1 byte at a time to the decoder to look for any bugs
	// when switching between decoding sequences
	lzma_ret ret = LZMA_OK;
	while (ret == LZMA_OK) {
		strm.avail_in = 1;
		ret = lzma_code(&strm, LZMA_RUN);
		if (strm.avail_out == 0) {
			checksum = lzma_crc32(output_buffer,
				(size_t)(strm.next_out - output_buffer),
				checksum);
			// No need to free output_buffer because it will
			// automatically be freed at the end of the test by
			// tuktest.
			output_buffer = tuktest_malloc(DECODE_CHUNK_SIZE);
			strm.next_out = output_buffer;
			strm.avail_out = DECODE_CHUNK_SIZE;
		}
	}

	assert_lzma_ret(ret, LZMA_STREAM_END);
	assert_uint_eq(strm.total_in, file_size);

	checksum = lzma_crc32(output_buffer,
			(size_t)(strm.next_out - output_buffer),
			checksum);
	assert_uint_eq(checksum, expected_crc);

	lzma_end(&strm);
}


static void
test_options(void)
{
	// Test NULL stream
	assert_lzma_ret(lzma_lzip_decoder(NULL, MEMLIMIT, 0),
			LZMA_PROG_ERROR);

	// Test invalid flags
	lzma_stream strm = LZMA_STREAM_INIT;
	assert_lzma_ret(lzma_lzip_decoder(&strm, MEMLIMIT, UINT32_MAX),
			LZMA_OPTIONS_ERROR);
	// Memlimit tests are done elsewhere
}


static void
test_v0_decode(void)
{
	// This tests if liblzma can decode lzip version 0 files.
	// lzip 1.17 and older can decompress this, but lzip 1.18
	// and newer can no longer decode these files.
	basic_lzip_decode("files/good-1-v0.lz", hello_world_crc);
}


static void
test_v1_decode(void)
{
	// This tests decoding a basic lzip v1 file
	basic_lzip_decode("files/good-1-v1.lz", hello_world_crc);
}


// Helper function to decode a good file with trailing bytes after
// the lzip stream
static void
trailing_helper(const char *src, const uint32_t expected_data_checksum,
		const uint32_t expected_trailing_checksum)
{
	size_t file_size;
	uint32_t checksum = 0;
	uint8_t *data = tuktest_file_from_srcdir(src, &file_size);
	lzma_stream strm = LZMA_STREAM_INIT;
	assert_lzma_ret(lzma_lzip_decoder(&strm, MEMLIMIT,
			LZMA_CONCATENATED), LZMA_OK);

	uint8_t *output_buffer = tuktest_malloc(DECODE_CHUNK_SIZE);

	strm.next_in = data;
	strm.next_out = output_buffer;
	strm.avail_in = file_size;
	strm.avail_out = DECODE_CHUNK_SIZE;

	lzma_ret ret = LZMA_OK;
	while (ret == LZMA_OK) {
		ret = lzma_code(&strm, LZMA_RUN);
		if (strm.avail_out == 0) {
			checksum = lzma_crc32(output_buffer,
				(size_t)(strm.next_out - output_buffer),
				checksum);
			// No need to free output_buffer because it will
			// automatically be freed at the end of the test by
			// tuktest.
			output_buffer = tuktest_malloc(DECODE_CHUNK_SIZE);
			strm.next_out = output_buffer;
			strm.avail_out = DECODE_CHUNK_SIZE;
		}
	}

	assert_lzma_ret(ret, LZMA_STREAM_END);
	assert_uint(strm.total_in, <, file_size);

	checksum = lzma_crc32(output_buffer,
			(size_t)(strm.next_out - output_buffer),
			checksum);

	assert_uint_eq(checksum, expected_data_checksum);

	// Trailing data should be readable from strm.next_in
	checksum = lzma_crc32(strm.next_in, strm.avail_in, 0);
	assert_uint_eq(checksum, expected_trailing_checksum);

	lzma_end(&strm);
}


// Helper function to decode a bad file and compare to returned error to
// what the caller expects
static void
decode_expect_error(const char *src, lzma_ret expected_error)
{
	lzma_stream strm = LZMA_STREAM_INIT;
	size_t file_size;
	uint8_t *data = tuktest_file_from_srcdir(src, &file_size);

	assert_lzma_ret(lzma_lzip_decoder(&strm, MEMLIMIT,
			LZMA_CONCATENATED), LZMA_OK);

	uint8_t output_buffer[DECODE_CHUNK_SIZE];

	strm.avail_in = file_size;
	strm.next_in = data;
	strm.avail_out = DECODE_CHUNK_SIZE;
	strm.next_out = output_buffer;

	lzma_ret ret = LZMA_OK;

	while (ret == LZMA_OK) {
		// Discard output since we are only looking for errors
		strm.next_out = output_buffer;
		strm.avail_out = DECODE_CHUNK_SIZE;
		if (strm.avail_in == 0)
			ret = lzma_code(&strm, LZMA_FINISH);
		else
			ret = lzma_code(&strm, LZMA_RUN);
	}

	assert_lzma_ret(ret, expected_error);
	lzma_end(&strm);
}


static void
test_v0_trailing(void)
{
	trailing_helper("files/good-1-v0-trailing-1.lz", hello_world_crc,
			trailing_garbage_crc);
}


static void
test_v1_trailing(void)
{
	trailing_helper("files/good-1-v1-trailing-1.lz", hello_world_crc,
			trailing_garbage_crc);

	// The second files/good-1-v1-trailing-2.lz will have the same
	// expected output and trailing output as
	// files/good-1-v1-trailing-1.lz, but this tests if the prefix
	// to the trailing data contains lzip magic bytes.
	// When this happens, the expected behavior is to silently ignore
	// the magic byte prefix and consume it from the input file.
	trailing_helper("files/good-1-v1-trailing-2.lz", hello_world_crc,
			trailing_garbage_crc);

	// Expect LZMA_BUF error if a file ends with the lzip magic bytes
	// but does not contain any data after
	decode_expect_error("files/bad-1-v1-trailing-magic.lz",
			LZMA_BUF_ERROR);
}


static void
test_concatentated(void)
{
	// First test a file with one v0 member and one v1 member
	// The first member should contain "Hello\n" and
	// the second member should contain "World!\n"

	lzma_stream strm = LZMA_STREAM_INIT;
	size_t file_size;
	uint8_t *v0_v1 = tuktest_file_from_srcdir("files/good-2-v0-v1.lz",
		&file_size);

	assert_lzma_ret(lzma_lzip_decoder(&strm, MEMLIMIT,
			LZMA_CONCATENATED), LZMA_OK);

	uint8_t output_buffer[DECODE_CHUNK_SIZE];

	strm.avail_in = file_size;
	strm.next_in = v0_v1;
	strm.avail_out = DECODE_CHUNK_SIZE;
	strm.next_out = output_buffer;

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);

	assert_uint_eq(strm.total_in, file_size);

	uint32_t checksum = lzma_crc32(output_buffer, strm.total_out, 0);
	assert_uint_eq(checksum, hello_world_crc);

	// The second file contains one v1 member and one v2 member
	uint8_t *v1_v0 = tuktest_file_from_srcdir("files/good-2-v1-v0.lz",
		&file_size);

	assert_lzma_ret(lzma_lzip_decoder(&strm, MEMLIMIT,
			LZMA_CONCATENATED), LZMA_OK);

	strm.avail_in = file_size;
	strm.next_in = v1_v0;
	strm.avail_out = DECODE_CHUNK_SIZE;
	strm.next_out = output_buffer;

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);

	assert_uint_eq(strm.total_in, file_size);
	checksum = lzma_crc32(output_buffer, strm.total_out, 0);
	assert_uint_eq(checksum, hello_world_crc);

	// The third file contains 2 v1 members
	uint8_t *v1_v1 = tuktest_file_from_srcdir("files/good-2-v1-v1.lz",
		&file_size);

	assert_lzma_ret(lzma_lzip_decoder(&strm, MEMLIMIT,
			LZMA_CONCATENATED), LZMA_OK);

	strm.avail_in = file_size;
	strm.next_in = v1_v1;
	strm.avail_out = DECODE_CHUNK_SIZE;
	strm.next_out = output_buffer;

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);

	assert_uint_eq(strm.total_in, file_size);
	checksum = lzma_crc32(output_buffer, strm.total_out, 0);
	assert_uint_eq(checksum, hello_world_crc);

	lzma_end(&strm);
}


static void
test_crc(void)
{
	// Test invalid checksum
	lzma_stream strm = LZMA_STREAM_INIT;
	size_t file_size;
	uint8_t *data = tuktest_file_from_srcdir("files/bad-1-v1-crc32.lz",
			&file_size);

	assert_lzma_ret(lzma_lzip_decoder(&strm, MEMLIMIT,
			LZMA_CONCATENATED), LZMA_OK);

	uint8_t output_buffer[DECODE_CHUNK_SIZE];

	strm.avail_in = file_size;
	strm.next_in = data;
	strm.avail_out = DECODE_CHUNK_SIZE;
	strm.next_out = output_buffer;

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_DATA_ERROR);

	// Test ignoring the checksum value - should decode successfully
	assert_lzma_ret(lzma_lzip_decoder(&strm, MEMLIMIT,
			LZMA_CONCATENATED | LZMA_IGNORE_CHECK), LZMA_OK);

	strm.avail_in = file_size;
	strm.next_in = data;
	strm.avail_out = DECODE_CHUNK_SIZE;
	strm.next_out = output_buffer;

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);
	assert_uint_eq(strm.total_in, file_size);

	// Test tell check
	assert_lzma_ret(lzma_lzip_decoder(&strm, MEMLIMIT,
			LZMA_CONCATENATED | LZMA_TELL_ANY_CHECK), LZMA_OK);

	strm.avail_in = file_size;
	strm.next_in = data;
	strm.avail_out = DECODE_CHUNK_SIZE;
	strm.next_out = output_buffer;

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_GET_CHECK);
	assert_uint_eq(lzma_get_check(&strm), LZMA_CHECK_CRC32);
	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_DATA_ERROR);
	lzma_end(&strm);
}


static void
test_invalid_magic_bytes(void)
{
	uint8_t lzip_id_string[] = { 0x4C, 0x5A, 0x49, 0x50 };
	lzma_stream strm = LZMA_STREAM_INIT;

	for (uint32_t i = 0; i < ARRAY_SIZE(lzip_id_string); i++) {
		// Corrupt magic bytes
		lzip_id_string[i] ^= 1;
		uint8_t output_buffer[DECODE_CHUNK_SIZE];

		assert_lzma_ret(lzma_lzip_decoder(&strm, MEMLIMIT, 0),
				LZMA_OK);

		strm.next_in = lzip_id_string;
		strm.avail_in = sizeof(lzip_id_string);
		strm.next_out = output_buffer;
		strm.avail_out = DECODE_CHUNK_SIZE;

		assert_lzma_ret(lzma_code(&strm, LZMA_RUN),
				LZMA_FORMAT_ERROR);

		// Reset magic bytes
		lzip_id_string[i] ^= 1;
	}

	lzma_end(&strm);
}


static void
test_invalid_version(void)
{
	// The file contains a version number that is not 0 or 1,
	// so it should cause an error
	decode_expect_error("files/unsupported-1-v234.lz",
			LZMA_OPTIONS_ERROR);
}


static void
test_invalid_dictionary_size(void)
{
	// First file has too small dictionary size field
	decode_expect_error("files/bad-1-v1-dict-1.lz", LZMA_DATA_ERROR);

	// Second file has too large dictionary size field
	decode_expect_error("files/bad-1-v1-dict-2.lz", LZMA_DATA_ERROR);
}


static void
test_invalid_uncomp_size(void)
{
	// Test invalid v0 lzip file uncomp size
	decode_expect_error("files/bad-1-v0-uncomp-size.lz",
			LZMA_DATA_ERROR);

	// Test invalid v1 lzip file uncomp size
	decode_expect_error("files/bad-1-v1-uncomp-size.lz",
			LZMA_DATA_ERROR);
}


static void
test_invalid_member_size(void)
{
	decode_expect_error("files/bad-1-v1-member-size.lz",
			LZMA_DATA_ERROR);
}


static void
test_invalid_memlimit(void)
{
	// A very low memlimit should prevent decoding.
	// Should be able to update the memlimit after failing
	size_t file_size;
	uint8_t *data = tuktest_file_from_srcdir("files/good-1-v1.lz",
			&file_size);

	uint8_t output_buffer[DECODE_CHUNK_SIZE];

	lzma_stream strm = LZMA_STREAM_INIT;

	assert_lzma_ret(lzma_lzip_decoder(&strm, 1, 0), LZMA_OK);

	strm.next_in = data;
	strm.avail_in = file_size;
	strm.next_out = output_buffer;
	strm.avail_out = DECODE_CHUNK_SIZE;

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_MEMLIMIT_ERROR);

	// Up the memlimit so decoding can continue.
	// First only increase by a small amount and expect an error
	assert_lzma_ret(lzma_memlimit_set(&strm, 100), LZMA_MEMLIMIT_ERROR);
	assert_lzma_ret(lzma_memlimit_set(&strm, MEMLIMIT), LZMA_OK);

	// Finish decoding
	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);

	assert_uint_eq(strm.total_in, file_size);
	uint32_t checksum = lzma_crc32(output_buffer, strm.total_out, 0);
	assert_uint_eq(checksum, hello_world_crc);

	lzma_end(&strm);
}
#endif


extern int
main(int argc, char **argv)
{
	tuktest_start(argc, argv);

#ifndef HAVE_LZIP_DECODER
	tuktest_early_skip("lzip decoder disabled");
#else
	tuktest_run(test_options);
	tuktest_run(test_v0_decode);
	tuktest_run(test_v1_decode);
	tuktest_run(test_v0_trailing);
	tuktest_run(test_v1_trailing);
	tuktest_run(test_concatentated);
	tuktest_run(test_crc);
	tuktest_run(test_invalid_magic_bytes);
	tuktest_run(test_invalid_version);
	tuktest_run(test_invalid_dictionary_size);
	tuktest_run(test_invalid_uncomp_size);
	tuktest_run(test_invalid_member_size);
	tuktest_run(test_invalid_memlimit);
	return tuktest_end();
#endif

}
