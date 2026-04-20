///////////////////////////////////////////////////////////////////////////////
//
/// \file       test_block_header.c
/// \brief      Tests Block Header coders
//
//  Authors:    Lasse Collin
//              Jia Tan
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "tests.h"


static lzma_options_lzma opt_lzma;


// Used in test_lzma_block_header_decode() between tests to ensure
// no artifacts are leftover in the block struct that could influence
// later tests.
#define RESET_BLOCK(block, buf) \
do { \
	lzma_filter *filters_ = (block).filters; \
	lzma_filters_free(filters_, NULL); \
	memzero((buf), sizeof((buf))); \
	memzero(&(block), sizeof(lzma_block)); \
	(block).filters = filters_; \
	(block).check = LZMA_CHECK_CRC32; \
} while (0);


#ifdef HAVE_ENCODERS
static lzma_filter filters_none[1] = {
	{
		.id = LZMA_VLI_UNKNOWN,
	},
};


static lzma_filter filters_one[2] = {
	{
		.id = LZMA_FILTER_LZMA2,
		.options = &opt_lzma,
	}, {
		.id = LZMA_VLI_UNKNOWN,
	}
};


// These filters are only used in test_lzma_block_header_decode()
// which only runs if encoders and decoders are configured.
#ifdef HAVE_DECODERS
static lzma_filter filters_four[5] = {
	{
		.id = LZMA_FILTER_X86,
		.options = NULL,
	}, {
		.id = LZMA_FILTER_X86,
		.options = NULL,
	}, {
		.id = LZMA_FILTER_X86,
		.options = NULL,
	}, {
		.id = LZMA_FILTER_LZMA2,
		.options = &opt_lzma,
	}, {
		.id = LZMA_VLI_UNKNOWN,
	}
};
#endif


static lzma_filter filters_five[6] = {
	{
		.id = LZMA_FILTER_X86,
		.options = NULL,
	}, {
		.id = LZMA_FILTER_X86,
		.options = NULL,
	}, {
		.id = LZMA_FILTER_X86,
		.options = NULL,
	}, {
		.id = LZMA_FILTER_X86,
		.options = NULL,
	}, {
		.id = LZMA_FILTER_LZMA2,
		.options = &opt_lzma,
	}, {
		.id = LZMA_VLI_UNKNOWN,
	}
};
#endif


static void
test_lzma_block_header_size(void)
{
#ifndef HAVE_ENCODERS
	assert_skip("Encoder support disabled");
#else
	if (!lzma_filter_encoder_is_supported(LZMA_FILTER_X86))
		assert_skip("x86 BCJ encoder is disabled");

	lzma_block block = {
		.version = 0,
		.filters = filters_one,
		.compressed_size = LZMA_VLI_UNKNOWN,
		.uncompressed_size = LZMA_VLI_UNKNOWN,
		.check = LZMA_CHECK_CRC32
	};

	// Test that all initial options are valid
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);
	assert_uint(block.header_size, >=, LZMA_BLOCK_HEADER_SIZE_MIN);
	assert_uint(block.header_size, <=, LZMA_BLOCK_HEADER_SIZE_MAX);
	assert_uint_eq(block.header_size % 4, 0);

	// Test invalid version number
	for (uint32_t i = 2; i < 20; i++) {
		block.version = i;
		assert_lzma_ret(lzma_block_header_size(&block),
				LZMA_OPTIONS_ERROR);
	}

	block.version = 1;

	// Test invalid compressed size
	block.compressed_size = 0;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_PROG_ERROR);

	block.compressed_size = LZMA_VLI_MAX + 1;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_PROG_ERROR);
	block.compressed_size = LZMA_VLI_UNKNOWN;

	// Test invalid uncompressed size
	block.uncompressed_size = LZMA_VLI_MAX + 1;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_PROG_ERROR);
	block.uncompressed_size = LZMA_VLI_MAX;

	// Test invalid filters
	block.filters = NULL;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_PROG_ERROR);

	block.filters = filters_none;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_PROG_ERROR);

	block.filters = filters_five;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_PROG_ERROR);

	block.filters = filters_one;

	// Test setting compressed_size to something valid
	block.compressed_size = 4096;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);
	assert_uint(block.header_size, >=, LZMA_BLOCK_HEADER_SIZE_MIN);
	assert_uint(block.header_size, <=, LZMA_BLOCK_HEADER_SIZE_MAX);
	assert_uint_eq(block.header_size % 4, 0);

	// Test setting uncompressed_size to something valid
	block.uncompressed_size = 4096;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);
	assert_uint(block.header_size, >=, LZMA_BLOCK_HEADER_SIZE_MIN);
	assert_uint(block.header_size, <=, LZMA_BLOCK_HEADER_SIZE_MAX);
	assert_uint_eq(block.header_size % 4, 0);

	// This should pass, but header_size will be an invalid value
	// because the total block size will not be able to fit in a valid
	// lzma_vli. This way a temporary value can be used to reserve
	// space for the header and later the actual value can be set.
	block.compressed_size = LZMA_VLI_MAX;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);
	assert_uint(block.header_size, >=, LZMA_BLOCK_HEADER_SIZE_MIN);
	assert_uint(block.header_size, <=, LZMA_BLOCK_HEADER_SIZE_MAX);
	assert_uint_eq(block.header_size % 4, 0);

	// Use an invalid value for a filter option. This should still pass
	// because the size of the LZMA2 properties is known by liblzma
	// without reading any of the options so it doesn't validate them.
	lzma_options_lzma bad_ops;
	assert_false(lzma_lzma_preset(&bad_ops, 1));
	bad_ops.pb = 0x1000;

	lzma_filter bad_filters[2] = {
		{
			.id = LZMA_FILTER_LZMA2,
			.options = &bad_ops
		},
		{
			.id = LZMA_VLI_UNKNOWN,
			.options = NULL
		}
	};

	block.filters = bad_filters;

	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);
	assert_uint(block.header_size, >=, LZMA_BLOCK_HEADER_SIZE_MIN);
	assert_uint(block.header_size, <=, LZMA_BLOCK_HEADER_SIZE_MAX);
	assert_uint_eq(block.header_size % 4, 0);

	// Use an invalid block option. The check type isn't stored in
	// the Block Header and so _header_size ignores it.
	block.check = INVALID_LZMA_CHECK_ID;
	block.ignore_check = false;

	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);
	assert_uint(block.header_size, >=, LZMA_BLOCK_HEADER_SIZE_MIN);
	assert_uint(block.header_size, <=, LZMA_BLOCK_HEADER_SIZE_MAX);
	assert_uint_eq(block.header_size % 4, 0);
#endif
}


static void
test_lzma_block_header_encode(void)
{
#if !defined(HAVE_ENCODERS) || !defined(HAVE_DECODERS)
	assert_skip("Encoder or decoder support disabled");
#else

	if (!lzma_filter_encoder_is_supported(LZMA_FILTER_X86)
                        || !lzma_filter_decoder_is_supported(LZMA_FILTER_X86))
                assert_skip("x86 BCJ encoder and/or decoder "
                                "is disabled");

	lzma_block block = {
		.version = 1,
		.filters = filters_one,
		.compressed_size = LZMA_VLI_UNKNOWN,
		.uncompressed_size = LZMA_VLI_UNKNOWN,
		.check = LZMA_CHECK_CRC32,
	};

	// Ensure all block options are valid before changes are tested
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);

	uint8_t out[LZMA_BLOCK_HEADER_SIZE_MAX];

	// Test invalid block version
	for (uint32_t i = 2; i < 20; i++) {
		block.version = i;
		assert_lzma_ret(lzma_block_header_encode(&block, out),
				LZMA_PROG_ERROR);
	}

	block.version = 1;

	// Test invalid header size (< min, > max, % 4 != 0)
	block.header_size = LZMA_BLOCK_HEADER_SIZE_MIN - 4;
	assert_lzma_ret(lzma_block_header_encode(&block, out),
			LZMA_PROG_ERROR);
	block.header_size = LZMA_BLOCK_HEADER_SIZE_MIN + 2;
	assert_lzma_ret(lzma_block_header_encode(&block, out),
			LZMA_PROG_ERROR);
	block.header_size = LZMA_BLOCK_HEADER_SIZE_MAX + 4;
	assert_lzma_ret(lzma_block_header_encode(&block, out),
			LZMA_PROG_ERROR);
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);

	// Test invalid compressed_size
	block.compressed_size = 0;
	assert_lzma_ret(lzma_block_header_encode(&block, out),
			LZMA_PROG_ERROR);
	block.compressed_size = LZMA_VLI_MAX + 1;
	assert_lzma_ret(lzma_block_header_encode(&block, out),
			LZMA_PROG_ERROR);

	// This test passes test_lzma_block_header_size, but should
	// fail here because there is not enough space to encode the
	// proper block size because the total size is too big to fit
	// in an lzma_vli
	block.compressed_size = LZMA_VLI_MAX;
	assert_lzma_ret(lzma_block_header_encode(&block, out),
			LZMA_PROG_ERROR);
	block.compressed_size = LZMA_VLI_UNKNOWN;

	// Test invalid uncompressed size
	block.uncompressed_size = LZMA_VLI_MAX + 1;
	assert_lzma_ret(lzma_block_header_encode(&block, out),
			LZMA_PROG_ERROR);
	block.uncompressed_size = LZMA_VLI_UNKNOWN;

	// Test invalid block check
	block.check = INVALID_LZMA_CHECK_ID;
	block.ignore_check = false;
	assert_lzma_ret(lzma_block_header_encode(&block, out),
			LZMA_PROG_ERROR);
	block.check = LZMA_CHECK_CRC32;

	// Test invalid filters
	block.filters = NULL;
	assert_lzma_ret(lzma_block_header_encode(&block, out),
			LZMA_PROG_ERROR);

	block.filters = filters_none;
	assert_lzma_ret(lzma_block_header_encode(&block, out),
			LZMA_PROG_ERROR);

	block.filters = filters_five;
	block.header_size = LZMA_BLOCK_HEADER_SIZE_MAX - 4;
	assert_lzma_ret(lzma_block_header_encode(&block, out),
			LZMA_PROG_ERROR);

	// Test valid encoding and verify bytes of block header.
	// More complicated tests for encoding headers are included
	// in test_lzma_block_header_decode.
	block.filters = filters_one;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);
	assert_lzma_ret(lzma_block_header_encode(&block, out), LZMA_OK);

	// First read block header size from out and verify
	// that it == (encoded size + 1) * 4
	uint32_t header_size = (out[0] + 1U) * 4;
	assert_uint_eq(header_size, block.header_size);

	// Next read block flags
	uint8_t flags = out[1];

	// Should have number of filters = 1
	assert_uint_eq((flags & 0x3) + 1, 1);

	// Bits 2-7 must be empty not set
	assert_uint_eq(flags & (0xFF - 0x3), 0);

	// Verify filter flags
	// Decode Filter ID
	lzma_vli filter_id = 0;
	size_t pos = 2;
	assert_lzma_ret(lzma_vli_decode(&filter_id, NULL, out,
			&pos, header_size), LZMA_OK);
	assert_uint_eq(filter_id, filters_one[0].id);

	// Decode Size of Properties
	lzma_vli prop_size = 0;
	assert_lzma_ret(lzma_vli_decode(&prop_size, NULL, out,
			&pos, header_size), LZMA_OK);

	// LZMA2 has 1 byte prop size
	assert_uint_eq(prop_size, 1);
	uint8_t expected_filter_props = 0;
	assert_lzma_ret(lzma_properties_encode(filters_one,
			&expected_filter_props), LZMA_OK);
	assert_uint_eq(out[pos], expected_filter_props);
	pos++;

	// Check null-padding
	for (size_t i = pos; i < header_size - 4; i++)
		assert_uint_eq(out[i], 0);

	// Check CRC32
	assert_uint_eq(read32le(&out[header_size - 4]), lzma_crc32(out,
			header_size - 4, 0));
#endif
}


#if defined(HAVE_ENCODERS) && defined(HAVE_DECODERS)
// Helper function to compare two lzma_block structures field by field
static void
compare_blocks(lzma_block *block_expected, lzma_block *block_actual)
{
	assert_uint_eq(block_actual->version, block_expected->version);
	assert_uint_eq(block_actual->compressed_size,
			block_expected->compressed_size);
	assert_uint_eq(block_actual->uncompressed_size,
			block_expected->uncompressed_size);
	assert_uint_eq(block_actual->check, block_expected->check);
	assert_uint_eq(block_actual->header_size, block_expected->header_size);

	// Compare filter IDs
	assert_true(block_expected->filters && block_actual->filters);
	lzma_filter expected_filter = block_expected->filters[0];
	uint32_t filter_count = 0;
	while (expected_filter.id != LZMA_VLI_UNKNOWN) {
		assert_uint_eq(block_actual->filters[filter_count].id,
				expected_filter.id);
		expected_filter = block_expected->filters[++filter_count];
	}

	assert_uint_eq(block_actual->filters[filter_count].id,
			LZMA_VLI_UNKNOWN);
}
#endif


static void
test_lzma_block_header_decode(void)
{
#if !defined(HAVE_ENCODERS) || !defined(HAVE_DECODERS)
	assert_skip("Encoder or decoder support disabled");
#else
	if (!lzma_filter_encoder_is_supported(LZMA_FILTER_X86)
                        || !lzma_filter_decoder_is_supported(LZMA_FILTER_X86))
                assert_skip("x86 BCJ encoder and/or decoder "
                                "is disabled");

	lzma_block block = {
		.filters = filters_one,
		.compressed_size = LZMA_VLI_UNKNOWN,
		.uncompressed_size = LZMA_VLI_UNKNOWN,
		.check = LZMA_CHECK_CRC32,
		.version = 0
	};

	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);

	// Encode block header with simple options
	uint8_t out[LZMA_BLOCK_HEADER_SIZE_MAX];
	assert_lzma_ret(lzma_block_header_encode(&block, out), LZMA_OK);

	// Decode block header and check that the options match
	lzma_filter decoded_filters[LZMA_FILTERS_MAX + 1];
	lzma_block decoded_block = {
		.version = 0,
		.filters = decoded_filters,
		.check = LZMA_CHECK_CRC32
	};
	decoded_block.header_size = lzma_block_header_size_decode(out[0]);

	assert_lzma_ret(lzma_block_header_decode(&decoded_block, NULL, out),
			LZMA_OK);
	compare_blocks(&block, &decoded_block);

	// Reset output buffer and decoded_block
	RESET_BLOCK(decoded_block, out);

	// Test with compressed size set
	block.compressed_size = 4096;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);
	assert_lzma_ret(lzma_block_header_encode(&block, out), LZMA_OK);
	decoded_block.header_size = lzma_block_header_size_decode(out[0]);
	assert_lzma_ret(lzma_block_header_decode(&decoded_block, NULL, out),
			LZMA_OK);
	compare_blocks(&block, &decoded_block);

	RESET_BLOCK(decoded_block, out);

	// Test with uncompressed size set
	block.uncompressed_size = 4096;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);
	assert_lzma_ret(lzma_block_header_encode(&block, out), LZMA_OK);
	decoded_block.header_size = lzma_block_header_size_decode(out[0]);
	assert_lzma_ret(lzma_block_header_decode(&decoded_block, NULL, out),
			LZMA_OK);
	compare_blocks(&block, &decoded_block);

	RESET_BLOCK(decoded_block, out);

	// Test with multiple filters
	block.filters = filters_four;
	assert_lzma_ret(lzma_block_header_size(&block), LZMA_OK);
	assert_lzma_ret(lzma_block_header_encode(&block, out), LZMA_OK);
	decoded_block.header_size = lzma_block_header_size_decode(out[0]);
	assert_lzma_ret(lzma_block_header_decode(&decoded_block, NULL, out),
			LZMA_OK);
	compare_blocks(&block, &decoded_block);

	lzma_filters_free(decoded_filters, NULL);

	// Test with too high version. The decoder will set it to a version
	// that it supports.
	decoded_block.version = 2;
	assert_lzma_ret(lzma_block_header_decode(&decoded_block, NULL, out),
			LZMA_OK);
	assert_uint_eq(decoded_block.version, 1);

	// Free the filters for the last time since all other cases should
	// result in an error.
	lzma_filters_free(decoded_filters, NULL);

	// Test bad check type
	decoded_block.check = INVALID_LZMA_CHECK_ID;
	assert_lzma_ret(lzma_block_header_decode(&decoded_block, NULL, out),
			LZMA_PROG_ERROR);
	decoded_block.check = LZMA_CHECK_CRC32;

	// Test bad check value
	out[decoded_block.header_size - 1] -= 10;
	assert_lzma_ret(lzma_block_header_decode(&decoded_block, NULL, out),
			LZMA_DATA_ERROR);
	out[decoded_block.header_size - 1] += 10;

	// Test non-NULL padding
	out[decoded_block.header_size - 5] = 1;

	// Recompute CRC32
	write32le(&out[decoded_block.header_size - 4], lzma_crc32(out,
			decoded_block.header_size - 4, 0));
	assert_lzma_ret(lzma_block_header_decode(&decoded_block, NULL, out),
			LZMA_OPTIONS_ERROR);

	// Test unsupported flags
	out[1] = 0xFF;

	// Recompute CRC32
	write32le(&out[decoded_block.header_size - 4], lzma_crc32(out,
			decoded_block.header_size - 4, 0));
	assert_lzma_ret(lzma_block_header_decode(&decoded_block, NULL, out),
			LZMA_OPTIONS_ERROR);
#endif
}


extern int
main(int argc, char **argv)
{
	tuktest_start(argc, argv);

	if (lzma_lzma_preset(&opt_lzma, 1))
		tuktest_error("lzma_lzma_preset() failed");

	tuktest_run(test_lzma_block_header_size);
	tuktest_run(test_lzma_block_header_encode);
	tuktest_run(test_lzma_block_header_decode);

	return tuktest_end();
}
