///////////////////////////////////////////////////////////////////////////////
//
/// \file       test_index_hash.c
/// \brief      Tests src/liblzma/common/index_hash.c API functions
///
/// \note       No test included for lzma_index_hash_end since it
///             would be trivial unless tested for memory leaks
///             with something like valgrind
//
//  Author:     Jia Tan
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "tests.h"

// Needed for UNPADDED_SIZE_MIN and UNPADDED_SIZE_MAX macro definitions
// and index_size and vli_ceil4 helper functions
#include "common/index.h"


static void
test_lzma_index_hash_init(void)
{
#ifndef HAVE_DECODERS
	assert_skip("Decoder support disabled");
#else
	// First test with NULL index_hash.
	// This should create a fresh index_hash.
	lzma_index_hash *index_hash = lzma_index_hash_init(NULL, NULL);
	assert_true(index_hash != NULL);

	// Next test with non-NULL index_hash.
	lzma_index_hash *second_hash = lzma_index_hash_init(index_hash, NULL);

	// It should not create a new index_hash pointer.
	// Instead it must just re-init the first index_hash.
	assert_true(index_hash == second_hash);

	lzma_index_hash_end(index_hash, NULL);
#endif
}


static void
test_lzma_index_hash_append(void)
{
#ifndef HAVE_DECODERS
	assert_skip("Decoder support disabled");
#else
	// Test all invalid parameters
	assert_lzma_ret(lzma_index_hash_append(NULL, 0, 0),
			LZMA_PROG_ERROR);

	// Test NULL index_hash
	assert_lzma_ret(lzma_index_hash_append(NULL, UNPADDED_SIZE_MIN,
			LZMA_VLI_MAX), LZMA_PROG_ERROR);

	// Test with invalid Unpadded Size
	lzma_index_hash *index_hash = lzma_index_hash_init(NULL, NULL);
	assert_true(index_hash != NULL);
	assert_lzma_ret(lzma_index_hash_append(index_hash,
			UNPADDED_SIZE_MIN - 1, LZMA_VLI_MAX),
			LZMA_PROG_ERROR);

	// Test with invalid Uncompressed Size
	assert_lzma_ret(lzma_index_hash_append(index_hash,
			UNPADDED_SIZE_MIN, LZMA_VLI_MAX + 1),
			LZMA_PROG_ERROR);

	// First append a Record describing a small Block.
	// This should succeed.
	assert_lzma_ret(lzma_index_hash_append(index_hash,
			UNPADDED_SIZE_MIN, 1), LZMA_OK);

	// Append another small Record.
	assert_lzma_ret(lzma_index_hash_append(index_hash,
			UNPADDED_SIZE_MIN, 1), LZMA_OK);

	// Append a Record that would cause the compressed size to grow
	// too big
	assert_lzma_ret(lzma_index_hash_append(index_hash,
			UNPADDED_SIZE_MAX, 1), LZMA_DATA_ERROR);

	lzma_index_hash_end(index_hash, NULL);
#endif
}


#if defined(HAVE_ENCODERS) && defined(HAVE_DECODERS)
// Fill an index_hash with unpadded and uncompressed VLIs
// by calling lzma_index_hash_append
static void
fill_index_hash(lzma_index_hash *index_hash, const lzma_vli *unpadded_sizes,
		const lzma_vli *uncomp_sizes, uint32_t block_count)
{
	for (uint32_t i = 0; i < block_count; ++i)
		assert_lzma_ret(lzma_index_hash_append(index_hash,
			unpadded_sizes[i], uncomp_sizes[i]), LZMA_OK);
}


// Set the contents of buf to the expected Index based on the
// .xz specification. This needs the unpadded and uncompressed VLIs
// to correctly create the Index.
static void
generate_index(uint8_t *buf, const lzma_vli *unpadded_sizes,
		const lzma_vli *uncomp_sizes, uint32_t block_count,
		size_t index_max_size)
{
	size_t in_pos = 0;
	size_t out_pos = 0;

	// First set Index Indicator
	buf[out_pos++] = INDEX_INDICATOR;

	// Next write out Number of Records
	assert_lzma_ret(lzma_vli_encode(block_count, &in_pos, buf,
			&out_pos, index_max_size), LZMA_STREAM_END);

	// Next write out each Record.
	// A Record consists of Unpadded Size and Uncompressed Size
	// written next to each other as VLIs.
	for (uint32_t i = 0; i < block_count; ++i) {
		in_pos = 0;
		assert_lzma_ret(lzma_vli_encode(unpadded_sizes[i], &in_pos,
			buf, &out_pos, index_max_size), LZMA_STREAM_END);
		in_pos = 0;
		assert_lzma_ret(lzma_vli_encode(uncomp_sizes[i], &in_pos,
			buf, &out_pos, index_max_size), LZMA_STREAM_END);
	}

	// Add Index Padding
	lzma_vli rounded_out_pos = vli_ceil4(out_pos);
	memzero(buf + out_pos, rounded_out_pos - out_pos);
	out_pos = rounded_out_pos;

	// Add the CRC32
	write32le(buf + out_pos, lzma_crc32(buf, out_pos, 0));
	out_pos += 4;

	assert_uint_eq(out_pos, index_max_size);
}
#endif


static void
test_lzma_index_hash_decode(void)
{
#if !defined(HAVE_ENCODERS) || !defined(HAVE_DECODERS)
	assert_skip("Encoder or decoder support disabled");
#else
	lzma_index_hash *index_hash = lzma_index_hash_init(NULL, NULL);
	assert_true(index_hash != NULL);

	size_t in_pos = 0;

	// Six valid values for the Unpadded Size fields in an Index
	const lzma_vli unpadded_sizes[6] = {
		UNPADDED_SIZE_MIN,
		1000,
		4000,
		8000,
		16000,
		32000
	};

	// Six valid values for the Uncompressed Size fields in an Index
	const lzma_vli uncomp_sizes[6] = {
		1,
		500,
		8000,
		20,
		1,
		500
	};

	// Add two Records to an index_hash
	fill_index_hash(index_hash, unpadded_sizes, uncomp_sizes, 2);

	const lzma_vli size_two_records = lzma_index_hash_size(index_hash);
	assert_uint(size_two_records, >, 0);
	uint8_t *index_two_records = tuktest_malloc(size_two_records);

	generate_index(index_two_records, unpadded_sizes, uncomp_sizes, 2,
			size_two_records);

	// First test for basic buffer size error
	in_pos = size_two_records + 1;
	assert_lzma_ret(lzma_index_hash_decode(index_hash,
			index_two_records, &in_pos,
			size_two_records), LZMA_BUF_ERROR);

	// Next test for invalid Index Indicator
	in_pos = 0;
	index_two_records[0] ^= 1;
	assert_lzma_ret(lzma_index_hash_decode(index_hash,
			index_two_records, &in_pos,
			size_two_records), LZMA_DATA_ERROR);
	index_two_records[0] ^= 1;

	// Next verify the index_hash as expected
	in_pos = 0;
	assert_lzma_ret(lzma_index_hash_decode(index_hash,
			index_two_records, &in_pos,
			size_two_records), LZMA_STREAM_END);

	// Next test an index_hash with three Records
	index_hash = lzma_index_hash_init(index_hash, NULL);
	fill_index_hash(index_hash, unpadded_sizes, uncomp_sizes, 3);

	const lzma_vli size_three_records = lzma_index_hash_size(
			index_hash);
	assert_uint(size_three_records, >, 0);
	uint8_t *index_three_records = tuktest_malloc(size_three_records);

	generate_index(index_three_records, unpadded_sizes, uncomp_sizes,
			3, size_three_records);

	in_pos = 0;
	assert_lzma_ret(lzma_index_hash_decode(index_hash,
			index_three_records, &in_pos,
			size_three_records), LZMA_STREAM_END);

	// Next test an index_hash with five Records
	index_hash = lzma_index_hash_init(index_hash, NULL);
	fill_index_hash(index_hash, unpadded_sizes, uncomp_sizes, 5);

	const lzma_vli size_five_records = lzma_index_hash_size(
			index_hash);
	assert_uint(size_five_records, >, 0);
	uint8_t *index_five_records = tuktest_malloc(size_five_records);

	generate_index(index_five_records, unpadded_sizes, uncomp_sizes, 5,
			size_five_records);

	// Instead of testing all input at once, give input
	// one byte at a time
	in_pos = 0;
	for (lzma_vli i = 0; i < size_five_records - 1; ++i) {
		assert_lzma_ret(lzma_index_hash_decode(index_hash,
				index_five_records, &in_pos, in_pos + 1),
				LZMA_OK);
	}

	// Last byte should return LZMA_STREAM_END
	assert_lzma_ret(lzma_index_hash_decode(index_hash,
			index_five_records, &in_pos,
			in_pos + 1), LZMA_STREAM_END);

	// Next test if the index_hash is given an incorrect Unpadded
	// Size. Should detect and report LZMA_DATA_ERROR
	index_hash = lzma_index_hash_init(index_hash, NULL);
	fill_index_hash(index_hash, unpadded_sizes, uncomp_sizes, 5);
	// The sixth Record will have an invalid Unpadded Size
	assert_lzma_ret(lzma_index_hash_append(index_hash,
			unpadded_sizes[5] + 1,
			uncomp_sizes[5]), LZMA_OK);

	const lzma_vli size_six_records = lzma_index_hash_size(
			index_hash);

	assert_uint(size_six_records, >, 0);
	uint8_t *index_six_records = tuktest_malloc(size_six_records);

	generate_index(index_six_records, unpadded_sizes, uncomp_sizes, 6,
			size_six_records);
	in_pos = 0;
	assert_lzma_ret(lzma_index_hash_decode(index_hash,
			index_six_records, &in_pos,
			size_six_records), LZMA_DATA_ERROR);

	// Next test if the Index is corrupt (invalid CRC32).
	// Should detect and report LZMA_DATA_ERROR
	index_hash = lzma_index_hash_init(index_hash, NULL);
	fill_index_hash(index_hash, unpadded_sizes, uncomp_sizes, 2);

	index_two_records[size_two_records - 1] ^= 1;

	in_pos = 0;
	assert_lzma_ret(lzma_index_hash_decode(index_hash,
			index_two_records, &in_pos,
			size_two_records), LZMA_DATA_ERROR);

	// Next test with Index and index_hash struct not matching
	// a Record
	index_hash = lzma_index_hash_init(index_hash, NULL);
	fill_index_hash(index_hash, unpadded_sizes, uncomp_sizes, 2);
	// Recalculate Index with invalid Unpadded Size
	const lzma_vli unpadded_sizes_invalid[2] = {
		unpadded_sizes[0],
		unpadded_sizes[1] + 1
	};

	generate_index(index_two_records, unpadded_sizes_invalid,
			uncomp_sizes, 2, size_two_records);

	in_pos = 0;
	assert_lzma_ret(lzma_index_hash_decode(index_hash,
			index_two_records, &in_pos,
			size_two_records), LZMA_DATA_ERROR);

	lzma_index_hash_end(index_hash, NULL);
#endif
}


static void
test_lzma_index_hash_size(void)
{
#ifndef HAVE_DECODERS
	assert_skip("Decoder support disabled");
#else
	lzma_index_hash *index_hash = lzma_index_hash_init(NULL, NULL);
	assert_true(index_hash != NULL);

	// First test empty index_hash
	// Expected size should be:
	// Index Indicator - 1 byte
	// Number of Records - 1 byte
	// List of Records - 0 bytes
	// Index Padding - 2 bytes
	// CRC32 - 4 bytes
	// Total - 8 bytes
	assert_uint_eq(lzma_index_hash_size(index_hash), 8);

	// Append a Record describing a small Block to the index_hash
	assert_lzma_ret(lzma_index_hash_append(index_hash,
			UNPADDED_SIZE_MIN, 1), LZMA_OK);

	// Expected size should be:
	// Index Indicator - 1 byte
	// Number of Records - 1 byte
	// List of Records - 2 bytes
	// Index Padding - 0 bytes
	// CRC32 - 4 bytes
	// Total - 8 bytes
	lzma_vli expected_size = 8;
	assert_uint_eq(lzma_index_hash_size(index_hash), expected_size);

	// Append additional small Record
	assert_lzma_ret(lzma_index_hash_append(index_hash,
			UNPADDED_SIZE_MIN, 1), LZMA_OK);

	// Expected size should be:
	// Index Indicator - 1 byte
	// Number of Records - 1 byte
	// List of Records - 4 bytes
	// Index Padding - 2 bytes
	// CRC32 - 4 bytes
	// Total - 12 bytes
	expected_size = 12;
	assert_uint_eq(lzma_index_hash_size(index_hash), expected_size);

	// Append a larger Record to the index_hash (3 bytes for each VLI)
	const lzma_vli three_byte_vli = 0x10000;
	assert_lzma_ret(lzma_index_hash_append(index_hash,
			three_byte_vli, three_byte_vli), LZMA_OK);

	// Expected size should be:
	// Index Indicator - 1 byte
	// Number of Records - 1 byte
	// List of Records - 10 bytes
	// Index Padding - 0 bytes
	// CRC32 - 4 bytes
	// Total - 16 bytes
	expected_size = 16;
	assert_uint_eq(lzma_index_hash_size(index_hash), expected_size);

	lzma_index_hash_end(index_hash, NULL);
#endif
}


extern int
main(int argc, char **argv)
{
	tuktest_start(argc, argv);
	tuktest_run(test_lzma_index_hash_init);
	tuktest_run(test_lzma_index_hash_append);
	tuktest_run(test_lzma_index_hash_decode);
	tuktest_run(test_lzma_index_hash_size);
	return tuktest_end();
}
