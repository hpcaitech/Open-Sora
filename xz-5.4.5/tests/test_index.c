///////////////////////////////////////////////////////////////////////////////
//
/// \file       test_index.c
/// \brief      Tests functions handling the lzma_index structure
///
/// \todo       Implement tests for lzma_file_info_decoder
//
//  Authors:    Jia Tan
//              Lasse Collin
//
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "tests.h"

// liblzma internal header file needed for:
// UNPADDED_SIZE_MIN
// UNPADDED_SIZE_MAX
// vli_ceil4
#include "common/index.h"


#define MEMLIMIT (UINT64_C(1) << 20)

static uint8_t *decode_buffer;
static size_t decode_buffer_size = 0;
static lzma_index *decode_test_index;


static void
test_lzma_index_memusage(void)
{
	// The return value from lzma_index_memusage is an approximation
	// of the amount of memory needed for lzma_index for a given
	// amount of Streams and Blocks. It will be an upperbound,
	// so this test will mostly sanity check and error check the
	// function.

	// The maximum number of Streams should be UINT32_MAX in the
	// current implementation even though the parameter is lzma_vli.
	assert_uint_eq(lzma_index_memusage((lzma_vli)UINT32_MAX + 1, 1),
			UINT64_MAX);

	// The maximum number of Blocks should be LZMA_VLI_MAX
	assert_uint_eq(lzma_index_memusage(1, LZMA_VLI_MAX), UINT64_MAX);

	// Number of Streams must be non-zero
	assert_uint_eq(lzma_index_memusage(0, 1), UINT64_MAX);

	// Number of Blocks CAN be zero
	assert_uint(lzma_index_memusage(1, 0), !=, UINT64_MAX);

	// Arbitrary values for Stream and Block should work without error
	// and should always increase
	uint64_t previous = 1;
	lzma_vli streams = 1;
	lzma_vli blocks = 1;

	// Test 100 different increasing values for Streams and Block
	for (int i = 0; i < 100; i++) {
		uint64_t current = lzma_index_memusage(streams, blocks);
		assert_uint(current, >, previous);
		previous = current;
		streams += 29;
		blocks += 107;
	}

	// Force integer overflow in calculation (should result in an error)
	assert_uint_eq(lzma_index_memusage(UINT32_MAX, LZMA_VLI_MAX),
			UINT64_MAX);
}


static void
test_lzma_index_memused(void)
{
	// Very similar to test_lzma_index_memusage above since
	// lzma_index_memused is essentially a wrapper for
	// lzma_index_memusage
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	// Test with empty Index
	assert_uint(lzma_index_memused(idx), <, UINT64_MAX);

	// Append small Blocks and then test again (should pass).
	for (lzma_vli i = 0; i < 10; i++)
		assert_lzma_ret(lzma_index_append(idx, NULL,
				UNPADDED_SIZE_MIN, 1), LZMA_OK);

	assert_uint(lzma_index_memused(idx), <, UINT64_MAX);

	lzma_index_end(idx, NULL);
}


static void
test_lzma_index_append(void)
{
	// Basic input-ouput test done here.
	// Less trivial tests for this function are done throughout
	// other tests.

	// First test with NULL lzma_index
	assert_lzma_ret(lzma_index_append(NULL, NULL, UNPADDED_SIZE_MIN,
			1), LZMA_PROG_ERROR);

	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	// Test with invalid Unpadded Size
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN - 1, 1), LZMA_PROG_ERROR);
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MAX + 1, 1), LZMA_PROG_ERROR);

	// Test with invalid Uncompressed Size
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MAX, LZMA_VLI_MAX + 1),
			LZMA_PROG_ERROR);

	// Test expected successful Block appends
	assert_lzma_ret(lzma_index_append(idx, NULL, UNPADDED_SIZE_MIN,
			1), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN * 2,
			2), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN * 3,
			3), LZMA_OK);

	lzma_index_end(idx, NULL);

	// Test compressed .xz file size growing too large. This also tests
	// a failing assert fixed in 68bda971bb8b666a009331455fcedb4e18d837a4.
	// Should result in LZMA_DATA_ERROR.
	idx = lzma_index_init(NULL);

	// The calculation for maximum unpadded size is to make room for the
	// second stream when lzma_index_cat() is called. The
	// 4 * LZMA_STREAM_HEADER_SIZE is for the header and footer of
	// both streams. The extra 24 bytes are for the size of the indexes
	// for both streams. This allows us to maximize the unpadded sum
	// during the lzma_index_append() call after the indexes have been
	// concatenated.
	assert_lzma_ret(lzma_index_append(idx, NULL, UNPADDED_SIZE_MAX
			- ((4 * LZMA_STREAM_HEADER_SIZE) + 24), 1), LZMA_OK);

	lzma_index *second = lzma_index_init(NULL);
	assert_true(second != NULL);

	assert_lzma_ret(lzma_index_cat(second, idx, NULL), LZMA_OK);

	assert_lzma_ret(lzma_index_append(second, NULL, UNPADDED_SIZE_MAX, 1),
			LZMA_DATA_ERROR);

	lzma_index_end(second, NULL);

	// Test uncompressed size growing too large.
	// Should result in LZMA_DATA_ERROR.
	idx = lzma_index_init(NULL);

	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN, LZMA_VLI_MAX), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN, 1), LZMA_DATA_ERROR);

	lzma_index_end(idx, NULL);

	// Currently not testing for error case when the size of the Index
	// grows too large to be stored. This was not practical to test for
	// since too many Blocks needed to be created to cause this.
}


static void
test_lzma_index_stream_flags(void)
{
	// Only trivial tests done here testing for basic functionality.
	// More in-depth testing for this function will be done in
	// test_lzma_index_checks.

	// Testing for NULL inputs
	assert_lzma_ret(lzma_index_stream_flags(NULL, NULL),
			LZMA_PROG_ERROR);

	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	assert_lzma_ret(lzma_index_stream_flags(idx, NULL),
			LZMA_PROG_ERROR);

	lzma_stream_flags stream_flags = {
		.version = 0,
		.backward_size = LZMA_BACKWARD_SIZE_MIN,
		.check = LZMA_CHECK_CRC32
	};

	assert_lzma_ret(lzma_index_stream_flags(idx, &stream_flags),
			LZMA_OK);

	lzma_index_end(idx, NULL);
}


static void
test_lzma_index_checks(void)
{
	// Tests should still pass, even if some of the check types
	// are disabled.
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	lzma_stream_flags stream_flags = {
		.version = 0,
		.backward_size = LZMA_BACKWARD_SIZE_MIN,
		.check = LZMA_CHECK_NONE
	};

	// First set the check type to None
	assert_lzma_ret(lzma_index_stream_flags(idx, &stream_flags),
			LZMA_OK);
	assert_uint_eq(lzma_index_checks(idx),
			UINT32_C(1) << LZMA_CHECK_NONE);

	// Set the check type to CRC32 and repeat
	stream_flags.check = LZMA_CHECK_CRC32;
	assert_lzma_ret(lzma_index_stream_flags(idx, &stream_flags),
			LZMA_OK);
	assert_uint_eq(lzma_index_checks(idx),
			UINT32_C(1) << LZMA_CHECK_CRC32);

	// Set the check type to CRC64 and repeat
	stream_flags.check = LZMA_CHECK_CRC64;
	assert_lzma_ret(lzma_index_stream_flags(idx, &stream_flags),
			LZMA_OK);
	assert_uint_eq(lzma_index_checks(idx),
			UINT32_C(1) << LZMA_CHECK_CRC64);

	// Set the check type to SHA256 and repeat
	stream_flags.check = LZMA_CHECK_SHA256;
	assert_lzma_ret(lzma_index_stream_flags(idx, &stream_flags),
			LZMA_OK);
	assert_uint_eq(lzma_index_checks(idx),
			UINT32_C(1) << LZMA_CHECK_SHA256);

	// Create second lzma_index and cat to first
	lzma_index *second = lzma_index_init(NULL);
	assert_true(second != NULL);

	// Set the check type to CRC32 for the second lzma_index
	stream_flags.check = LZMA_CHECK_CRC32;
	assert_lzma_ret(lzma_index_stream_flags(second, &stream_flags),
			LZMA_OK);

	assert_uint_eq(lzma_index_checks(second),
			UINT32_C(1) << LZMA_CHECK_CRC32);

	assert_lzma_ret(lzma_index_cat(idx, second, NULL), LZMA_OK);

	// Index should now have both CRC32 and SHA256
	assert_uint_eq(lzma_index_checks(idx),
			(UINT32_C(1) << LZMA_CHECK_CRC32) |
			(UINT32_C(1) << LZMA_CHECK_SHA256));

	// Change the check type of the second Stream to SHA256
	stream_flags.check = LZMA_CHECK_SHA256;
	assert_lzma_ret(lzma_index_stream_flags(idx, &stream_flags),
			LZMA_OK);

	// Index should now have only SHA256
	assert_uint_eq(lzma_index_checks(idx),
			UINT32_C(1) << LZMA_CHECK_SHA256);

	// Test with a third Stream
	lzma_index *third = lzma_index_init(NULL);
	assert_true(third != NULL);

	stream_flags.check = LZMA_CHECK_CRC64;
	assert_lzma_ret(lzma_index_stream_flags(third, &stream_flags),
			LZMA_OK);

	assert_uint_eq(lzma_index_checks(third),
			UINT32_C(1) << LZMA_CHECK_CRC64);

	assert_lzma_ret(lzma_index_cat(idx, third, NULL), LZMA_OK);

	// Index should now have CRC64 and SHA256
	assert_uint_eq(lzma_index_checks(idx),
			(UINT32_C(1) << LZMA_CHECK_CRC64) |
			(UINT32_C(1) << LZMA_CHECK_SHA256));

	lzma_index_end(idx, NULL);
}


static void
test_lzma_index_stream_padding(void)
{
	// Test NULL lzma_index
	assert_lzma_ret(lzma_index_stream_padding(NULL, 0),
			LZMA_PROG_ERROR);

	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	// Test Stream Padding not a multiple of 4
	assert_lzma_ret(lzma_index_stream_padding(idx, 3),
			LZMA_PROG_ERROR);

	// Test Stream Padding too large
	assert_lzma_ret(lzma_index_stream_padding(idx, LZMA_VLI_MAX - 3),
			LZMA_DATA_ERROR);

	// Test Stream Padding valid
	assert_lzma_ret(lzma_index_stream_padding(idx, 0x1000),
			LZMA_OK);
	assert_lzma_ret(lzma_index_stream_padding(idx, 4),
			LZMA_OK);
	assert_lzma_ret(lzma_index_stream_padding(idx, 0),
			LZMA_OK);

	// Test Stream Padding causing the file size to grow too large
	assert_lzma_ret(lzma_index_append(idx, NULL,
			LZMA_VLI_MAX - 0x1000, 1), LZMA_OK);
	assert_lzma_ret(lzma_index_stream_padding(idx, 0x1000),
			LZMA_DATA_ERROR);

	lzma_index_end(idx, NULL);
}


static void
test_lzma_index_stream_count(void)
{
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	assert_uint_eq(lzma_index_stream_count(idx), 1);

	// Appending Blocks should not change the Stream count value
	assert_lzma_ret(lzma_index_append(idx, NULL, UNPADDED_SIZE_MIN,
			1), LZMA_OK);

	assert_uint_eq(lzma_index_stream_count(idx), 1);

	// Test with multiple Streams
	for (uint32_t i = 0; i < 100; i++) {
		lzma_index *idx_cat = lzma_index_init(NULL);
		assert_true(idx != NULL);
		assert_lzma_ret(lzma_index_cat(idx, idx_cat, NULL), LZMA_OK);
		assert_uint_eq(lzma_index_stream_count(idx), i + 2);
	}

	lzma_index_end(idx, NULL);
}


static void
test_lzma_index_block_count(void)
{
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	assert_uint_eq(lzma_index_block_count(idx), 0);

	const uint32_t iterations = 0x1000;
	for (uint32_t i = 0; i < iterations; i++) {
		assert_lzma_ret(lzma_index_append(idx, NULL,
				UNPADDED_SIZE_MIN, 1), LZMA_OK);
		assert_uint_eq(lzma_index_block_count(idx), i + 1);
	}

	// Create new lzma_index with a few Blocks
	lzma_index *second = lzma_index_init(NULL);
	assert_true(second != NULL);

	assert_lzma_ret(lzma_index_append(second, NULL,
			UNPADDED_SIZE_MIN, 1), LZMA_OK);
	assert_lzma_ret(lzma_index_append(second, NULL,
			UNPADDED_SIZE_MIN, 1), LZMA_OK);
	assert_lzma_ret(lzma_index_append(second, NULL,
			UNPADDED_SIZE_MIN, 1), LZMA_OK);

	assert_uint_eq(lzma_index_block_count(second), 3);

	// Concatenate the lzma_indexes together and the result should have
	// the sum of the two individual counts.
	assert_lzma_ret(lzma_index_cat(idx, second, NULL), LZMA_OK);
	assert_uint_eq(lzma_index_block_count(idx), iterations + 3);

	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN, 1), LZMA_OK);

	assert_uint_eq(lzma_index_block_count(idx), iterations + 4);

	lzma_index_end(idx, NULL);
}


static void
test_lzma_index_size(void)
{
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	// Base size should be:
	// 1 byte Index Indicator
	// 1 byte Number of Records
	// 0 bytes Records
	// 2 bytes Index Padding
	// 4 bytes CRC32
	// Total: 8 bytes
	assert_uint_eq(lzma_index_size(idx), 8);

	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN, 1), LZMA_OK);

	// New size should be:
	// 1 byte Index Indicator
	// 1 byte Number of Records
	// 2 bytes Records
	// 0 bytes Index Padding
	// 4 bytes CRC32
	// Total: 8 bytes
	assert_uint_eq(lzma_index_size(idx), 8);

	assert_lzma_ret(lzma_index_append(idx, NULL,
			LZMA_VLI_MAX / 4, LZMA_VLI_MAX / 4), LZMA_OK);

	// New size should be:
	// 1 byte Index Indicator
	// 1 byte Number of Records
	// 20 bytes Records
	// 2 bytes Index Padding
	// 4 bytes CRC32
	// Total: 28 bytes
	assert_uint_eq(lzma_index_size(idx), 28);

	lzma_index_end(idx, NULL);
}


static void
test_lzma_index_stream_size(void)
{
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	// Stream size calculated by:
	// Size of Stream Header (12 bytes)
	// Size of all Blocks
	// Size of the Index
	// Size of the Stream Footer (12 bytes)

	// First test with empty Index
	// Stream size should be:
	// Size of Stream Header - 12 bytes
	// Size of all Blocks - 0 bytes
	// Size of Index - 8 bytes
	// Size of Stream Footer - 12 bytes
	// Total: 32 bytes
	assert_uint_eq(lzma_index_stream_size(idx), 32);

	// Next, append a few Blocks and retest
	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 1), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 1), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 1), LZMA_OK);

	// Stream size should be:
	// Size of Stream Header - 12 bytes
	// Size of all Blocks - 3000 bytes
	// Size of Index - 16 bytes
	// Size of Stream Footer - 12 bytes
	// Total: 3040 bytes
	assert_uint_eq(lzma_index_stream_size(idx), 3040);

	lzma_index *second = lzma_index_init(NULL);
	assert_true(second != NULL);

	assert_uint_eq(lzma_index_stream_size(second), 32);
	assert_lzma_ret(lzma_index_append(second, NULL, 1000, 1), LZMA_OK);

	// Stream size should be:
	// Size of Stream Header - 12 bytes
	// Size of all Blocks - 1000 bytes
	// Size of Index - 12 bytes
	// Size of Stream Footer - 12 bytes
	// Total: 1036 bytes
	assert_uint_eq(lzma_index_stream_size(second), 1036);

	assert_lzma_ret(lzma_index_cat(idx, second, NULL), LZMA_OK);

	// Stream size should be:
	// Size of Stream Header - 12 bytes
	// Size of all Blocks - 4000 bytes
	// Size of Index - 20 bytes
	// Size of Stream Footer - 12 bytes
	// Total: 4044 bytes
	assert_uint_eq(lzma_index_stream_size(idx), 4044);

	lzma_index_end(idx, NULL);
}


static void
test_lzma_index_total_size(void)
{
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	// First test empty lzma_index.
	// Result should be 0 since no Blocks have been added.
	assert_uint_eq(lzma_index_total_size(idx), 0);

	// Add a few Blocks and retest after each append
	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 1), LZMA_OK);
	assert_uint_eq(lzma_index_total_size(idx), 1000);

	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 1), LZMA_OK);
	assert_uint_eq(lzma_index_total_size(idx), 2000);

	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 1), LZMA_OK);
	assert_uint_eq(lzma_index_total_size(idx), 3000);

	// Create second lzma_index and append Blocks to it.
	lzma_index *second = lzma_index_init(NULL);
	assert_true(second != NULL);

	assert_uint_eq(lzma_index_total_size(second), 0);

	assert_lzma_ret(lzma_index_append(second, NULL, 100, 1), LZMA_OK);
	assert_uint_eq(lzma_index_total_size(second), 100);

	assert_lzma_ret(lzma_index_append(second, NULL, 100, 1), LZMA_OK);
	assert_uint_eq(lzma_index_total_size(second), 200);

	// Concatenate the Streams together
	assert_lzma_ret(lzma_index_cat(idx, second, NULL), LZMA_OK);

	// The resulting total size should be the size of all Blocks
	// from both Streams
	assert_uint_eq(lzma_index_total_size(idx), 3200);

	lzma_index_end(idx, NULL);
}


static void
test_lzma_index_file_size(void)
{
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	// Should be the same as test_lzma_index_stream_size with
	// only one Stream and no Stream Padding.
	assert_uint_eq(lzma_index_file_size(idx), 32);

	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 1), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 1), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 1), LZMA_OK);

	assert_uint_eq(lzma_index_file_size(idx), 3040);

	// Next add Stream Padding
	assert_lzma_ret(lzma_index_stream_padding(idx, 1000),
			LZMA_OK);

	assert_uint_eq(lzma_index_file_size(idx), 4040);

	// Create second lzma_index.
	// Very similar to test_lzma_index_stream_size, but
	// the values should include the headers of the second Stream.
	lzma_index *second = lzma_index_init(NULL);
	assert_true(second != NULL);

	assert_lzma_ret(lzma_index_append(second, NULL, 1000, 1), LZMA_OK);
	assert_uint_eq(lzma_index_stream_size(second), 1036);

	assert_lzma_ret(lzma_index_cat(idx, second, NULL), LZMA_OK);

	// .xz file size should be:
	// Size of 2 Stream Headers - 12 * 2 bytes
	// Size of all Blocks - 3000 + 1000 bytes
	// Size of 2 Indexes - 16 + 12 bytes
	// Size of Stream Padding - 1000 bytes
	// Size of 2 Stream Footers - 12 * 2 bytes
	// Total: 5076 bytes
	assert_uint_eq(lzma_index_file_size(idx), 5076);

	lzma_index_end(idx, NULL);
}


static void
test_lzma_index_uncompressed_size(void)
{
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	// Empty lzma_index should have 0 uncompressed .xz file size.
	assert_uint_eq(lzma_index_uncompressed_size(idx), 0);

	// Append a few small Blocks
	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 1), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 10), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 100), LZMA_OK);

	assert_uint_eq(lzma_index_uncompressed_size(idx), 111);

	// Create another lzma_index
	lzma_index *second = lzma_index_init(NULL);
	assert_true(second != NULL);

	// Append a few small Blocks
	assert_lzma_ret(lzma_index_append(second, NULL, 1000, 2), LZMA_OK);
	assert_lzma_ret(lzma_index_append(second, NULL, 1000, 20), LZMA_OK);
	assert_lzma_ret(lzma_index_append(second, NULL, 1000, 200), LZMA_OK);

	assert_uint_eq(lzma_index_uncompressed_size(second), 222);

	// Concatenate second lzma_index to first
	assert_lzma_ret(lzma_index_cat(idx, second, NULL), LZMA_OK);

	// New uncompressed .xz file size should be the sum of the two Streams
	assert_uint_eq(lzma_index_uncompressed_size(idx), 333);

	// Append one more Block to the lzma_index and ensure that
	// it is properly updated
	assert_lzma_ret(lzma_index_append(idx, NULL, 1000, 111), LZMA_OK);
	assert_uint_eq(lzma_index_uncompressed_size(idx), 444);

	lzma_index_end(idx, NULL);
}


static void
test_lzma_index_iter_init(void)
{
	// Testing basic init functionality.
	// The init function should call rewind on the iterator.
	lzma_index *first = lzma_index_init(NULL);
	assert_true(first != NULL);

	lzma_index *second = lzma_index_init(NULL);
	assert_true(second != NULL);

	lzma_index *third = lzma_index_init(NULL);
	assert_true(third != NULL);

	assert_lzma_ret(lzma_index_cat(first, second, NULL), LZMA_OK);
	assert_lzma_ret(lzma_index_cat(first, third, NULL), LZMA_OK);

	lzma_index_iter iter;
	lzma_index_iter_init(&iter, first);

	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_STREAM));
	assert_uint_eq(iter.stream.number, 1);
	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_STREAM));
	assert_uint_eq(iter.stream.number, 2);

	lzma_index_iter_init(&iter, first);

	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_STREAM));
	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_STREAM));
	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_STREAM));
	assert_uint_eq(iter.stream.number, 3);

	lzma_index_end(first, NULL);
}


static void
test_lzma_index_iter_rewind(void)
{
	lzma_index *first = lzma_index_init(NULL);
	assert_true(first != NULL);

	lzma_index_iter iter;
	lzma_index_iter_init(&iter, first);

	// Append 3 Blocks and iterate over each. This is to test
	// the LZMA_INDEX_ITER_BLOCK mode.
	for (uint32_t i = 0; i < 3; i++) {
		assert_lzma_ret(lzma_index_append(first, NULL,
				UNPADDED_SIZE_MIN, 1), LZMA_OK);
		assert_false(lzma_index_iter_next(&iter,
				LZMA_INDEX_ITER_BLOCK));
		assert_uint_eq(iter.block.number_in_file, i + 1);
	}

	// Rewind back to the beginning and iterate over the Blocks again
	lzma_index_iter_rewind(&iter);

	// Should be able to re-iterate over the Blocks again.
	for (uint32_t i = 0; i < 3; i++) {
		assert_false(lzma_index_iter_next(&iter,
				LZMA_INDEX_ITER_BLOCK));
		assert_uint_eq(iter.block.number_in_file, i + 1);
	}

	// Next concatenate two more lzma_indexes, iterate over them,
	// rewind, and iterate over them again. This is to test
	// the LZMA_INDEX_ITER_STREAM mode.
	lzma_index *second = lzma_index_init(NULL);
	assert_true(second != NULL);

	lzma_index *third = lzma_index_init(NULL);
	assert_true(third != NULL);

	assert_lzma_ret(lzma_index_cat(first, second, NULL), LZMA_OK);
	assert_lzma_ret(lzma_index_cat(first, third, NULL), LZMA_OK);

	assert_false(lzma_index_iter_next(&iter,
			LZMA_INDEX_ITER_STREAM));
	assert_false(lzma_index_iter_next(&iter,
			LZMA_INDEX_ITER_STREAM));

	assert_uint_eq(iter.stream.number, 3);

	lzma_index_iter_rewind(&iter);

	for (uint32_t i = 0; i < 3; i++) {
		assert_false(lzma_index_iter_next(&iter,
				LZMA_INDEX_ITER_STREAM));
		assert_uint_eq(iter.stream.number, i + 1);
	}

	lzma_index_end(first, NULL);
}


static void
test_lzma_index_iter_next(void)
{
	lzma_index *first = lzma_index_init(NULL);
	assert_true(first != NULL);

	lzma_index_iter iter;
	lzma_index_iter_init(&iter, first);

	// First test bad mode values
	for (uint32_t i = LZMA_INDEX_ITER_NONEMPTY_BLOCK + 1; i < 100; i++)
		assert_true(lzma_index_iter_next(&iter, i));

	// Test iterating over Blocks
	assert_lzma_ret(lzma_index_append(first, NULL,
			UNPADDED_SIZE_MIN, 1), LZMA_OK);
	assert_lzma_ret(lzma_index_append(first, NULL,
			UNPADDED_SIZE_MIN * 2, 10), LZMA_OK);
	assert_lzma_ret(lzma_index_append(first, NULL,
			UNPADDED_SIZE_MIN * 3, 100), LZMA_OK);

	// For Blocks, need to verify:
	// - number_in_file (overall Block number)
	// - compressed_file_offset
	// - uncompressed_file_offset
	// - number_in_stream (Block number relative to current Stream)
	// - compressed_stream_offset
	// - uncompressed_stream_offset
	// - uncompressed_size
	// - unpadded_size
	// - total_size

	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_BLOCK));

	// Verify Block data stored correctly
	assert_uint_eq(iter.block.number_in_file, 1);

	// Should start right after the Stream Header
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE);
	assert_uint_eq(iter.block.uncompressed_file_offset, 0);
	assert_uint_eq(iter.block.number_in_stream, 1);
	assert_uint_eq(iter.block.compressed_stream_offset,
			LZMA_STREAM_HEADER_SIZE);
	assert_uint_eq(iter.block.uncompressed_stream_offset, 0);
	assert_uint_eq(iter.block.unpadded_size, UNPADDED_SIZE_MIN);
	assert_uint_eq(iter.block.total_size, vli_ceil4(UNPADDED_SIZE_MIN));

	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_BLOCK));

	// Verify Block data stored correctly
	assert_uint_eq(iter.block.number_in_file, 2);
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE +
			vli_ceil4(UNPADDED_SIZE_MIN));
	assert_uint_eq(iter.block.uncompressed_file_offset, 1);
	assert_uint_eq(iter.block.number_in_stream, 2);
	assert_uint_eq(iter.block.compressed_stream_offset,
			LZMA_STREAM_HEADER_SIZE +
			vli_ceil4(UNPADDED_SIZE_MIN));
	assert_uint_eq(iter.block.uncompressed_stream_offset, 1);
	assert_uint_eq(iter.block.unpadded_size, UNPADDED_SIZE_MIN * 2);
	assert_uint_eq(iter.block.total_size, vli_ceil4(UNPADDED_SIZE_MIN * 2));

	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_BLOCK));

	// Verify Block data stored correctly
	assert_uint_eq(iter.block.number_in_file, 3);
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE +
			vli_ceil4(UNPADDED_SIZE_MIN) +
			vli_ceil4(UNPADDED_SIZE_MIN * 2));
	assert_uint_eq(iter.block.uncompressed_file_offset, 11);
	assert_uint_eq(iter.block.number_in_stream, 3);
	assert_uint_eq(iter.block.compressed_stream_offset,
			LZMA_STREAM_HEADER_SIZE +
			vli_ceil4(UNPADDED_SIZE_MIN) +
			vli_ceil4(UNPADDED_SIZE_MIN * 2));
	assert_uint_eq(iter.block.uncompressed_stream_offset, 11);
	assert_uint_eq(iter.block.unpadded_size, UNPADDED_SIZE_MIN * 3);
	assert_uint_eq(iter.block.total_size,
			vli_ceil4(UNPADDED_SIZE_MIN * 3));

	// Only three Blocks were added, so this should return true
	assert_true(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_BLOCK));

	const lzma_vli second_stream_compressed_start =
			LZMA_STREAM_HEADER_SIZE * 2 +
			vli_ceil4(UNPADDED_SIZE_MIN) +
			vli_ceil4(UNPADDED_SIZE_MIN * 2) +
			vli_ceil4(UNPADDED_SIZE_MIN * 3) +
			lzma_index_size(first);
	const lzma_vli second_stream_uncompressed_start = 1 + 10 + 100;

	// Test iterating over Streams.
	// The second Stream will have 0 Blocks
	lzma_index *second = lzma_index_init(NULL);
	assert_true(second != NULL);

	// Set Stream Flags for Stream 2
	lzma_stream_flags flags = {
		.version = 0,
		.backward_size = LZMA_BACKWARD_SIZE_MIN,
		.check = LZMA_CHECK_CRC32
	};

	assert_lzma_ret(lzma_index_stream_flags(second, &flags), LZMA_OK);

	// The Second stream will have 8 bytes of Stream Padding
	assert_lzma_ret(lzma_index_stream_padding(second, 8), LZMA_OK);

	const lzma_vli second_stream_index_size = lzma_index_size(second);

	// The third Stream will have 2 Blocks
	lzma_index *third = lzma_index_init(NULL);
	assert_true(third != NULL);

	assert_lzma_ret(lzma_index_append(third, NULL, 32, 20), LZMA_OK);
	assert_lzma_ret(lzma_index_append(third, NULL, 64, 40), LZMA_OK);

	const lzma_vli third_stream_index_size = lzma_index_size(third);

	assert_lzma_ret(lzma_index_cat(first, second, NULL), LZMA_OK);
	assert_lzma_ret(lzma_index_cat(first, third, NULL), LZMA_OK);

	// For Streams, need to verify:
	// - flags (Stream Flags)
	// - number (Stream count)
	// - block_count
	// - compressed_offset
	// - uncompressed_offset
	// - compressed_size
	// - uncompressed_size
	// - padding (Stream Padding)
	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_STREAM));

	// Verify Stream
	assert_uint_eq(iter.stream.flags->backward_size,
			LZMA_BACKWARD_SIZE_MIN);
	assert_uint_eq(iter.stream.flags->check, LZMA_CHECK_CRC32);
	assert_uint_eq(iter.stream.number, 2);
	assert_uint_eq(iter.stream.block_count, 0);
	assert_uint_eq(iter.stream.compressed_offset,
			second_stream_compressed_start);
	assert_uint_eq(iter.stream.uncompressed_offset,
			second_stream_uncompressed_start);
	assert_uint_eq(iter.stream.compressed_size,
			LZMA_STREAM_HEADER_SIZE * 2 +
			second_stream_index_size);
	assert_uint_eq(iter.stream.uncompressed_size, 0);
	assert_uint_eq(iter.stream.padding, 8);

	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_STREAM));

	// Verify Stream
	const lzma_vli third_stream_compressed_start =
			second_stream_compressed_start +
			LZMA_STREAM_HEADER_SIZE * 2 +
			8 + // Stream padding
			second_stream_index_size;
	const lzma_vli third_stream_uncompressed_start =
			second_stream_uncompressed_start;

	assert_uint_eq(iter.stream.number, 3);
	assert_uint_eq(iter.stream.block_count, 2);
	assert_uint_eq(iter.stream.compressed_offset,
			third_stream_compressed_start);
	assert_uint_eq(iter.stream.uncompressed_offset,
			third_stream_uncompressed_start);
	assert_uint_eq(iter.stream.compressed_size,
			LZMA_STREAM_HEADER_SIZE * 2 +
			96 + // Total compressed size
			third_stream_index_size);
	assert_uint_eq(iter.stream.uncompressed_size, 60);
	assert_uint_eq(iter.stream.padding, 0);

	assert_true(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_STREAM));

	// Even after a failing call to next with ITER_STREAM mode,
	// should still be able to iterate over the 2 Blocks in
	// Stream 3.
	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_BLOCK));

	// Verify both Blocks

	// Next call to iterate Block should return true because the
	// first Block can already be read from the LZMA_INDEX_ITER_STREAM
	// call.
	assert_true(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_BLOCK));

	// Rewind to test LZMA_INDEX_ITER_ANY
	lzma_index_iter_rewind(&iter);

	// Iterate past the first three Blocks
	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_ANY));
	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_ANY));
	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_ANY));

	// Iterate past the next Stream
	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_ANY));

	// Iterate past the next Stream
	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_ANY));
	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_ANY));

	// Last call should fail
	assert_true(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_ANY));

	// Rewind to test LZMA_INDEX_ITER_NONEMPTY_BLOCK
	lzma_index_iter_rewind(&iter);

	// Iterate past the first three Blocks
	assert_false(lzma_index_iter_next(&iter,
			LZMA_INDEX_ITER_NONEMPTY_BLOCK));
	assert_false(lzma_index_iter_next(&iter,
			LZMA_INDEX_ITER_NONEMPTY_BLOCK));
	assert_false(lzma_index_iter_next(&iter,
			LZMA_INDEX_ITER_NONEMPTY_BLOCK));

	// Skip past the next Stream which has no Blocks.
	// We will get to the first Block of the third Stream.
	assert_false(lzma_index_iter_next(&iter,
			LZMA_INDEX_ITER_NONEMPTY_BLOCK));

	// Iterate past the second (the last) Block in the third Stream
	assert_false(lzma_index_iter_next(&iter,
			LZMA_INDEX_ITER_NONEMPTY_BLOCK));

	// Last call should fail since there is nothing left to iterate over.
	assert_true(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_ANY));

	lzma_index_end(first, NULL);
}


static void
test_lzma_index_iter_locate(void)
{
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	lzma_index_iter iter;
	lzma_index_iter_init(&iter, idx);

	// Cannot locate anything from an empty Index.
	assert_true(lzma_index_iter_locate(&iter, 0));
	assert_true(lzma_index_iter_locate(&iter, 555));

	// One empty Record: nothing is found since there's no uncompressed
	// data.
	assert_lzma_ret(lzma_index_append(idx, NULL, 16, 0), LZMA_OK);
	assert_true(lzma_index_iter_locate(&iter, 0));

	// Non-empty Record and we can find something.
	assert_lzma_ret(lzma_index_append(idx, NULL, 32, 5), LZMA_OK);
	assert_false(lzma_index_iter_locate(&iter, 0));
	assert_uint_eq(iter.block.total_size, 32);
	assert_uint_eq(iter.block.uncompressed_size, 5);
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE + 16);
	assert_uint_eq(iter.block.uncompressed_file_offset, 0);

	// Still cannot find anything past the end.
	assert_true(lzma_index_iter_locate(&iter, 5));

	// Add the third Record.
	assert_lzma_ret(lzma_index_append(idx, NULL, 40, 11), LZMA_OK);

	assert_false(lzma_index_iter_locate(&iter, 0));
	assert_uint_eq(iter.block.total_size, 32);
	assert_uint_eq(iter.block.uncompressed_size, 5);
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE + 16);
	assert_uint_eq(iter.block.uncompressed_file_offset, 0);

	assert_false(lzma_index_iter_next(&iter, LZMA_INDEX_ITER_BLOCK));
	assert_uint_eq(iter.block.total_size, 40);
	assert_uint_eq(iter.block.uncompressed_size, 11);
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE + 16 + 32);
	assert_uint_eq(iter.block.uncompressed_file_offset, 5);

	assert_false(lzma_index_iter_locate(&iter, 2));
	assert_uint_eq(iter.block.total_size, 32);
	assert_uint_eq(iter.block.uncompressed_size, 5);
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE + 16);
	assert_uint_eq(iter.block.uncompressed_file_offset, 0);

	assert_false(lzma_index_iter_locate(&iter, 5));
	assert_uint_eq(iter.block.total_size, 40);
	assert_uint_eq(iter.block.uncompressed_size, 11);
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE + 16 + 32);
	assert_uint_eq(iter.block.uncompressed_file_offset, 5);

	assert_false(lzma_index_iter_locate(&iter, 5 + 11 - 1));
	assert_uint_eq(iter.block.total_size, 40);
	assert_uint_eq(iter.block.uncompressed_size, 11);
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE + 16 + 32);
	assert_uint_eq(iter.block.uncompressed_file_offset, 5);

	assert_true(lzma_index_iter_locate(&iter, 5 + 11));
	assert_true(lzma_index_iter_locate(&iter, 5 + 15));

	// Large Index
	lzma_index_end(idx, NULL);
	idx = lzma_index_init(NULL);
	assert_true(idx != NULL);
	lzma_index_iter_init(&iter, idx);

	for (uint32_t n = 4; n <= 4 * 5555; n += 4)
		assert_lzma_ret(lzma_index_append(idx, NULL, n + 8, n),
				LZMA_OK);

	assert_uint_eq(lzma_index_block_count(idx), 5555);

	// First Record
	assert_false(lzma_index_iter_locate(&iter, 0));
	assert_uint_eq(iter.block.total_size, 4 + 8);
	assert_uint_eq(iter.block.uncompressed_size, 4);
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE);
	assert_uint_eq(iter.block.uncompressed_file_offset, 0);

	assert_false(lzma_index_iter_locate(&iter, 3));
	assert_uint_eq(iter.block.total_size, 4 + 8);
	assert_uint_eq(iter.block.uncompressed_size, 4);
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE);
	assert_uint_eq(iter.block.uncompressed_file_offset, 0);

	// Second Record
	assert_false(lzma_index_iter_locate(&iter, 4));
	assert_uint_eq(iter.block.total_size, 2 * 4 + 8);
	assert_uint_eq(iter.block.uncompressed_size, 2 * 4);
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE + 4 + 8);
	assert_uint_eq(iter.block.uncompressed_file_offset, 4);

	// Last Record
	assert_false(lzma_index_iter_locate(
			&iter, lzma_index_uncompressed_size(idx) - 1));
	assert_uint_eq(iter.block.total_size, 4 * 5555 + 8);
	assert_uint_eq(iter.block.uncompressed_size, 4 * 5555);
	assert_uint_eq(iter.block.compressed_file_offset,
			lzma_index_total_size(idx)
			+ LZMA_STREAM_HEADER_SIZE - 4 * 5555 - 8);
	assert_uint_eq(iter.block.uncompressed_file_offset,
			lzma_index_uncompressed_size(idx) - 4 * 5555);

	// Allocation chunk boundaries. See INDEX_GROUP_SIZE in
	// liblzma/common/index.c.
	const uint32_t group_multiple = 256 * 4;
	const uint32_t radius = 8;
	const uint32_t start = group_multiple - radius;
	lzma_vli ubase = 0;
	lzma_vli tbase = 0;
	uint32_t n;
	for (n = 1; n < start; ++n) {
		ubase += n * 4;
		tbase += n * 4 + 8;
	}

	while (n < start + 2 * radius) {
		assert_false(lzma_index_iter_locate(&iter, ubase + n * 4));

		assert_uint_eq(iter.block.compressed_file_offset,
				tbase + n * 4 + 8
				+ LZMA_STREAM_HEADER_SIZE);
		assert_uint_eq(iter.block.uncompressed_file_offset,
				ubase + n * 4);

		tbase += n * 4 + 8;
		ubase += n * 4;
		++n;

		assert_uint_eq(iter.block.total_size, n * 4 + 8);
		assert_uint_eq(iter.block.uncompressed_size, n * 4);
	}

	// Do it also backwards.
	while (n > start) {
		assert_false(lzma_index_iter_locate(
				&iter, ubase + (n - 1) * 4));

		assert_uint_eq(iter.block.total_size, n * 4 + 8);
		assert_uint_eq(iter.block.uncompressed_size, n * 4);

		--n;
		tbase -= n * 4 + 8;
		ubase -= n * 4;

		assert_uint_eq(iter.block.compressed_file_offset,
				tbase + n * 4 + 8
				+ LZMA_STREAM_HEADER_SIZE);
		assert_uint_eq(iter.block.uncompressed_file_offset,
				ubase + n * 4);
	}

	// Test locating in concatenated Index.
	lzma_index_end(idx, NULL);
	idx = lzma_index_init(NULL);
	assert_true(idx != NULL);
	lzma_index_iter_init(&iter, idx);
	for (n = 0; n < group_multiple; ++n)
		assert_lzma_ret(lzma_index_append(idx, NULL, 8, 0),
				LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL, 16, 1), LZMA_OK);
	assert_false(lzma_index_iter_locate(&iter, 0));
	assert_uint_eq(iter.block.total_size, 16);
	assert_uint_eq(iter.block.uncompressed_size, 1);
	assert_uint_eq(iter.block.compressed_file_offset,
			LZMA_STREAM_HEADER_SIZE + group_multiple * 8);
	assert_uint_eq(iter.block.uncompressed_file_offset, 0);

	lzma_index_end(idx, NULL);
}


static void
test_lzma_index_cat(void)
{
	// Most complex tests for this function are done in other tests.
	// This will mostly test basic functionality.

	lzma_index *dest = lzma_index_init(NULL);
	assert_true(dest != NULL);

	lzma_index *src = lzma_index_init(NULL);
	assert_true(src != NULL);

	// First test NULL dest or src
	assert_lzma_ret(lzma_index_cat(NULL, NULL, NULL), LZMA_PROG_ERROR);
	assert_lzma_ret(lzma_index_cat(dest, NULL, NULL), LZMA_PROG_ERROR);
	assert_lzma_ret(lzma_index_cat(NULL, src, NULL), LZMA_PROG_ERROR);

	// Check for uncompressed size overflow
	assert_lzma_ret(lzma_index_append(dest, NULL,
			(UNPADDED_SIZE_MAX / 2) + 1, 1), LZMA_OK);
	assert_lzma_ret(lzma_index_append(src, NULL,
			(UNPADDED_SIZE_MAX / 2) + 1, 1), LZMA_OK);
	assert_lzma_ret(lzma_index_cat(dest, src, NULL), LZMA_DATA_ERROR);

	// Check for compressed size overflow
	lzma_index_end(src, NULL);
	lzma_index_end(dest, NULL);

	dest = lzma_index_init(NULL);
	assert_true(dest != NULL);

	src = lzma_index_init(NULL);
	assert_true(src != NULL);

	assert_lzma_ret(lzma_index_append(dest, NULL,
			UNPADDED_SIZE_MIN, LZMA_VLI_MAX - 1), LZMA_OK);
	assert_lzma_ret(lzma_index_append(src, NULL,
			UNPADDED_SIZE_MIN, LZMA_VLI_MAX - 1), LZMA_OK);
	assert_lzma_ret(lzma_index_cat(dest, src, NULL), LZMA_DATA_ERROR);

	lzma_index_end(dest, NULL);
	lzma_index_end(src, NULL);
}


// Helper function for test_lzma_index_dup().
static bool
index_is_equal(const lzma_index *a, const lzma_index *b)
{
	// Compare only the Stream and Block sizes and offsets.
	lzma_index_iter ra, rb;
	lzma_index_iter_init(&ra, a);
	lzma_index_iter_init(&rb, b);

	while (true) {
		bool reta = lzma_index_iter_next(&ra, LZMA_INDEX_ITER_ANY);
		bool retb = lzma_index_iter_next(&rb, LZMA_INDEX_ITER_ANY);

		// If both iterators finish at the same time, then the Indexes
		// are identical.
		if (reta)
			return retb;

		if (ra.stream.number != rb.stream.number
				|| ra.stream.block_count
					!= rb.stream.block_count
				|| ra.stream.compressed_offset
					!= rb.stream.compressed_offset
				|| ra.stream.uncompressed_offset
					!= rb.stream.uncompressed_offset
				|| ra.stream.compressed_size
					!= rb.stream.compressed_size
				|| ra.stream.uncompressed_size
					!= rb.stream.uncompressed_size
				|| ra.stream.padding
					!= rb.stream.padding)
			return false;

		if (ra.stream.block_count == 0)
			continue;

		if (ra.block.number_in_file != rb.block.number_in_file
				|| ra.block.compressed_file_offset
					!= rb.block.compressed_file_offset
				|| ra.block.uncompressed_file_offset
					!= rb.block.uncompressed_file_offset
				|| ra.block.number_in_stream
					!= rb.block.number_in_stream
				|| ra.block.compressed_stream_offset
					!= rb.block.compressed_stream_offset
				|| ra.block.uncompressed_stream_offset
					!= rb.block.uncompressed_stream_offset
				|| ra.block.uncompressed_size
					!= rb.block.uncompressed_size
				|| ra.block.unpadded_size
					!= rb.block.unpadded_size
				|| ra.block.total_size
					!= rb.block.total_size)
			return false;
	}
}


// Allocator that succeeds for the first two allocation but fails the rest.
static void *
my_alloc(void *opaque, size_t a, size_t b)
{
	(void)opaque;

	static unsigned count = 0;
	if (++count > 2)
		return NULL;

	return malloc(a * b);
}

static const lzma_allocator test_index_dup_alloc = { &my_alloc, NULL, NULL };


static void
test_lzma_index_dup(void)
{
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	// Test for the bug fix 21515d79d778b8730a434f151b07202d52a04611:
	// liblzma: Fix lzma_index_dup() for empty Streams.
	assert_lzma_ret(lzma_index_stream_padding(idx, 4), LZMA_OK);
	lzma_index *copy = lzma_index_dup(idx, NULL);
	assert_true(copy != NULL);
	assert_true(index_is_equal(idx, copy));
	lzma_index_end(copy, NULL);

	// Test for the bug fix 3bf857edfef51374f6f3fffae3d817f57d3264a0:
	// liblzma: Fix a memory leak in error path of lzma_index_dup().
	// Use Valgrind to see that there are no leaks.
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN, 10), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN * 2, 100), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN * 3, 1000), LZMA_OK);

	assert_true(lzma_index_dup(idx, &test_index_dup_alloc) == NULL);

	// Test a few streams and blocks
	lzma_index *second = lzma_index_init(NULL);
	assert_true(second != NULL);

	assert_lzma_ret(lzma_index_stream_padding(second, 16), LZMA_OK);

	lzma_index *third = lzma_index_init(NULL);
	assert_true(third != NULL);

	assert_lzma_ret(lzma_index_append(third, NULL,
			UNPADDED_SIZE_MIN * 10, 40), LZMA_OK);
	assert_lzma_ret(lzma_index_append(third, NULL,
			UNPADDED_SIZE_MIN * 20, 400), LZMA_OK);
	assert_lzma_ret(lzma_index_append(third, NULL,
			UNPADDED_SIZE_MIN * 30, 4000), LZMA_OK);

	assert_lzma_ret(lzma_index_cat(idx, second, NULL), LZMA_OK);
	assert_lzma_ret(lzma_index_cat(idx, third, NULL), LZMA_OK);

	copy = lzma_index_dup(idx, NULL);
	assert_true(copy != NULL);
	assert_true(index_is_equal(idx, copy));

	lzma_index_end(copy, NULL);
	lzma_index_end(idx, NULL);
}

#if defined(HAVE_ENCODERS) && defined(HAVE_DECODERS)
static void
verify_index_buffer(const lzma_index *idx, const uint8_t *buffer,
		const size_t buffer_size)
{
	lzma_index_iter iter;
	lzma_index_iter_init(&iter, idx);

	size_t buffer_pos = 0;

	// Verify Index Indicator
	assert_uint_eq(buffer[buffer_pos++], 0);

	// Get Number of Records
	lzma_vli number_of_records = 0;
	lzma_vli block_count = 0;
	assert_lzma_ret(lzma_vli_decode(&number_of_records, NULL, buffer,
			&buffer_pos, buffer_size), LZMA_OK);

	while (!lzma_index_iter_next(&iter, LZMA_INDEX_ITER_ANY)) {
		// Verify each Record (Unpadded Size, then Uncompressed Size).
		// Verify Unpadded Size.
		lzma_vli unpadded_size, uncompressed_size;
		assert_lzma_ret(lzma_vli_decode(&unpadded_size,
				NULL, buffer, &buffer_pos,
				buffer_size), LZMA_OK);
		assert_uint_eq(unpadded_size,
				iter.block.unpadded_size);

		// Verify Uncompressed Size
		assert_lzma_ret(lzma_vli_decode(&uncompressed_size,
				NULL, buffer, &buffer_pos,
				buffer_size), LZMA_OK);
		assert_uint_eq(uncompressed_size,
				iter.block.uncompressed_size);

		block_count++;
	}

	// Verify Number of Records
	assert_uint_eq(number_of_records, block_count);

	// Verify Index Padding
	for (; buffer_pos % 4 != 0; buffer_pos++)
		assert_uint_eq(buffer[buffer_pos], 0);

	// Verify CRC32
	uint32_t crc32 = lzma_crc32(buffer, buffer_pos, 0);
	assert_uint_eq(read32le(buffer + buffer_pos), crc32);
}


// In a few places the Index size is needed as a size_t but lzma_index_size()
// returns lzma_vli.
static size_t
get_index_size(const lzma_index *idx)
{
	const lzma_vli size = lzma_index_size(idx);
	assert_uint(size, <, SIZE_MAX);
	return (size_t)size;
}
#endif


static void
test_lzma_index_encoder(void)
{
#if !defined(HAVE_ENCODERS) || !defined(HAVE_DECODERS)
	assert_skip("Encoder or decoder support disabled");
#else
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	lzma_stream strm = LZMA_STREAM_INIT;

	// First do basic NULL checks
	assert_lzma_ret(lzma_index_encoder(NULL, NULL), LZMA_PROG_ERROR);
	assert_lzma_ret(lzma_index_encoder(&strm, NULL), LZMA_PROG_ERROR);
	assert_lzma_ret(lzma_index_encoder(NULL, idx), LZMA_PROG_ERROR);

	// Append three small Blocks
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN, 10), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN * 2, 100), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN * 3, 1000), LZMA_OK);

	// Encode this lzma_index into a buffer
	size_t buffer_size = get_index_size(idx);
	uint8_t *buffer = tuktest_malloc(buffer_size);

	assert_lzma_ret(lzma_index_encoder(&strm, idx), LZMA_OK);

	strm.avail_out = buffer_size;
	strm.next_out = buffer;

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);
	assert_uint_eq(strm.avail_out, 0);

	lzma_end(&strm);

	verify_index_buffer(idx, buffer, buffer_size);

	// Test with multiple Streams concatenated into 1 Index
	lzma_index *second = lzma_index_init(NULL);
	assert_true(second != NULL);

	// Include 1 Block
	assert_lzma_ret(lzma_index_append(second, NULL,
			UNPADDED_SIZE_MIN * 4, 20), LZMA_OK);

	// Include Stream Padding
	assert_lzma_ret(lzma_index_stream_padding(second, 16), LZMA_OK);

	assert_lzma_ret(lzma_index_cat(idx, second, NULL), LZMA_OK);
	buffer_size = get_index_size(idx);
	buffer = tuktest_malloc(buffer_size);
	assert_lzma_ret(lzma_index_encoder(&strm, idx), LZMA_OK);

	strm.avail_out = buffer_size;
	strm.next_out = buffer;

	assert_lzma_ret(lzma_code(&strm, LZMA_FINISH), LZMA_STREAM_END);
	assert_uint_eq(strm.avail_out, 0);

	verify_index_buffer(idx, buffer, buffer_size);

	lzma_index_end(idx, NULL);
	lzma_end(&strm);
#endif
}

static void
generate_index_decode_buffer(void)
{
#ifdef HAVE_ENCODERS
	decode_test_index = lzma_index_init(NULL);
	if (decode_test_index == NULL)
		return;

	// Add 4 Blocks
	for (uint32_t i = 1; i < 5; i++)
		if (lzma_index_append(decode_test_index, NULL,
				0x1000 * i, 0x100 * i) != LZMA_OK)
			return;

	size_t size = lzma_index_size(decode_test_index);
	decode_buffer = tuktest_malloc(size);

	if (lzma_index_buffer_encode(decode_test_index,
			decode_buffer, &decode_buffer_size, size) != LZMA_OK)
		decode_buffer_size = 0;
#endif
}


#ifdef HAVE_DECODERS
static void
decode_index(const uint8_t *buffer, const size_t size, lzma_stream *strm,
		lzma_ret expected_error)
{
	strm->avail_in = size;
	strm->next_in = buffer;
	assert_lzma_ret(lzma_code(strm, LZMA_FINISH), expected_error);
}
#endif


static void
test_lzma_index_decoder(void)
{
#ifndef HAVE_DECODERS
	assert_skip("Decoder support disabled");
#else
	if (decode_buffer_size == 0)
		assert_skip("Could not initialize decode test buffer");

	lzma_stream strm = LZMA_STREAM_INIT;

	assert_lzma_ret(lzma_index_decoder(NULL, NULL, MEMLIMIT),
			LZMA_PROG_ERROR);
	assert_lzma_ret(lzma_index_decoder(&strm, NULL, MEMLIMIT),
			LZMA_PROG_ERROR);
	assert_lzma_ret(lzma_index_decoder(NULL, &decode_test_index,
			MEMLIMIT), LZMA_PROG_ERROR);

	// Do actual decode
	lzma_index *idx;
	assert_lzma_ret(lzma_index_decoder(&strm, &idx, MEMLIMIT),
			LZMA_OK);

	decode_index(decode_buffer, decode_buffer_size, &strm,
			LZMA_STREAM_END);

	// Compare results with expected
	assert_true(index_is_equal(decode_test_index, idx));

	lzma_index_end(idx, NULL);

	// Test again with too low memory limit
	assert_lzma_ret(lzma_index_decoder(&strm, &idx, 0), LZMA_OK);

	decode_index(decode_buffer, decode_buffer_size, &strm,
			LZMA_MEMLIMIT_ERROR);

	uint8_t *corrupt_buffer = tuktest_malloc(decode_buffer_size);
	memcpy(corrupt_buffer, decode_buffer, decode_buffer_size);

	assert_lzma_ret(lzma_index_decoder(&strm, &idx, MEMLIMIT),
			LZMA_OK);

	// First corrupt the Index Indicator
	corrupt_buffer[0] ^= 1;
	decode_index(corrupt_buffer, decode_buffer_size, &strm,
			LZMA_DATA_ERROR);
	corrupt_buffer[0] ^= 1;

	// Corrupt something in the middle of Index
	corrupt_buffer[decode_buffer_size / 2] ^= 1;
	assert_lzma_ret(lzma_index_decoder(&strm, &idx, MEMLIMIT),
			LZMA_OK);
	decode_index(corrupt_buffer, decode_buffer_size, &strm,
			LZMA_DATA_ERROR);
	corrupt_buffer[decode_buffer_size / 2] ^= 1;

	// Corrupt CRC32
	corrupt_buffer[decode_buffer_size - 1] ^= 1;
	assert_lzma_ret(lzma_index_decoder(&strm, &idx, MEMLIMIT),
			LZMA_OK);
	decode_index(corrupt_buffer, decode_buffer_size, &strm,
			LZMA_DATA_ERROR);
	corrupt_buffer[decode_buffer_size - 1] ^= 1;

	// Corrupt Index Padding by setting it to non-zero
	corrupt_buffer[decode_buffer_size - 5] ^= 1;
	assert_lzma_ret(lzma_index_decoder(&strm, &idx, MEMLIMIT),
			LZMA_OK);
	decode_index(corrupt_buffer, decode_buffer_size, &strm,
			LZMA_DATA_ERROR);
	corrupt_buffer[decode_buffer_size - 1] ^= 1;

	lzma_end(&strm);
#endif
}


static void
test_lzma_index_buffer_encode(void)
{
#if !defined(HAVE_ENCODERS) || !defined(HAVE_DECODERS)
	assert_skip("Encoder or decoder support disabled");
#else
	// More simple test than test_lzma_index_encoder() because
	// currently lzma_index_buffer_encode() is mostly a wrapper
	// around lzma_index_encoder() anyway.
	lzma_index *idx = lzma_index_init(NULL);
	assert_true(idx != NULL);

	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN, 10), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN * 2, 100), LZMA_OK);
	assert_lzma_ret(lzma_index_append(idx, NULL,
			UNPADDED_SIZE_MIN * 3, 1000), LZMA_OK);

	size_t buffer_size = get_index_size(idx);
	uint8_t *buffer = tuktest_malloc(buffer_size);
	size_t out_pos = 1;

	// First test bad arguments
	assert_lzma_ret(lzma_index_buffer_encode(NULL, NULL, NULL, 0),
			LZMA_PROG_ERROR);
	assert_lzma_ret(lzma_index_buffer_encode(idx, NULL, NULL, 0),
			LZMA_PROG_ERROR);
	assert_lzma_ret(lzma_index_buffer_encode(idx, buffer, NULL, 0),
			LZMA_PROG_ERROR);
	assert_lzma_ret(lzma_index_buffer_encode(idx, buffer, &out_pos,
			0), LZMA_PROG_ERROR);
	out_pos = 0;
	assert_lzma_ret(lzma_index_buffer_encode(idx, buffer, &out_pos,
			1), LZMA_BUF_ERROR);

	// Do encoding
	assert_lzma_ret(lzma_index_buffer_encode(idx, buffer, &out_pos,
			buffer_size), LZMA_OK);
	assert_uint_eq(out_pos, buffer_size);

	// Validate results
	verify_index_buffer(idx, buffer, buffer_size);

	lzma_index_end(idx, NULL);
#endif
}


static void
test_lzma_index_buffer_decode(void)
{
#ifndef HAVE_DECODERS
	assert_skip("Decoder support disabled");
#else
	if (decode_buffer_size == 0)
		assert_skip("Could not initialize decode test buffer");

	// Simple test since test_lzma_index_decoder() covers most of the
	// lzma_index_buffer_decode() code anyway.

	// First test NULL checks
	assert_lzma_ret(lzma_index_buffer_decode(NULL, NULL, NULL, NULL,
			NULL, 0), LZMA_PROG_ERROR);

	lzma_index *idx;
	uint64_t memlimit = MEMLIMIT;
	size_t in_pos = 0;

	assert_lzma_ret(lzma_index_buffer_decode(&idx, NULL, NULL, NULL,
			NULL, 0), LZMA_PROG_ERROR);

	assert_lzma_ret(lzma_index_buffer_decode(&idx, &memlimit, NULL,
			NULL, NULL, 0), LZMA_PROG_ERROR);

	assert_lzma_ret(lzma_index_buffer_decode(&idx, &memlimit, NULL,
			decode_buffer, NULL, 0), LZMA_PROG_ERROR);

	assert_lzma_ret(lzma_index_buffer_decode(&idx, &memlimit, NULL,
			decode_buffer, NULL, 0), LZMA_PROG_ERROR);

	assert_lzma_ret(lzma_index_buffer_decode(&idx, &memlimit, NULL,
			decode_buffer, &in_pos, 0), LZMA_DATA_ERROR);

	in_pos = 1;
	assert_lzma_ret(lzma_index_buffer_decode(&idx, &memlimit, NULL,
			decode_buffer, &in_pos, 0), LZMA_PROG_ERROR);
	in_pos = 0;

	// Test expected successful decode
	assert_lzma_ret(lzma_index_buffer_decode(&idx, &memlimit, NULL,
			decode_buffer, &in_pos, decode_buffer_size), LZMA_OK);

	assert_true(index_is_equal(decode_test_index, idx));

	lzma_index_end(idx, NULL);

	// Test too small memlimit
	in_pos = 0;
	memlimit = 1;
	assert_lzma_ret(lzma_index_buffer_decode(&idx, &memlimit, NULL,
			decode_buffer, &in_pos, decode_buffer_size),
			LZMA_MEMLIMIT_ERROR);
	assert_uint(memlimit, >, 1);
	assert_uint(memlimit, <, MEMLIMIT);
#endif
}


extern int
main(int argc, char **argv)
{
	tuktest_start(argc, argv);
	generate_index_decode_buffer();
	tuktest_run(test_lzma_index_memusage);
	tuktest_run(test_lzma_index_memused);
	tuktest_run(test_lzma_index_append);
	tuktest_run(test_lzma_index_stream_flags);
	tuktest_run(test_lzma_index_checks);
	tuktest_run(test_lzma_index_stream_padding);
	tuktest_run(test_lzma_index_stream_count);
	tuktest_run(test_lzma_index_block_count);
	tuktest_run(test_lzma_index_size);
	tuktest_run(test_lzma_index_stream_size);
	tuktest_run(test_lzma_index_total_size);
	tuktest_run(test_lzma_index_file_size);
	tuktest_run(test_lzma_index_uncompressed_size);
	tuktest_run(test_lzma_index_iter_init);
	tuktest_run(test_lzma_index_iter_rewind);
	tuktest_run(test_lzma_index_iter_next);
	tuktest_run(test_lzma_index_iter_locate);
	tuktest_run(test_lzma_index_cat);
	tuktest_run(test_lzma_index_dup);
	tuktest_run(test_lzma_index_encoder);
	tuktest_run(test_lzma_index_decoder);
	tuktest_run(test_lzma_index_buffer_encode);
	tuktest_run(test_lzma_index_buffer_decode);
	lzma_index_end(decode_test_index, NULL);
	return tuktest_end();
}
