///////////////////////////////////////////////////////////////////////////////
//
/// \file       test_filter_flags.c
/// \brief      Tests Filter Flags coders
//
//  Authors:    Jia Tan
//              Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "tests.h"

// FIXME: This is from src/liblzma/common/common.h but it cannot be
// included here. This constant is needed in only a few files, perhaps
// move it to some other internal header or create a new one?
#define LZMA_FILTER_RESERVED_START (LZMA_VLI_C(1) << 62)


#if defined(HAVE_ENCODERS)
// No tests are run without encoders, so init the global filters
// only when the encoders are enabled.
static lzma_filter lzma1_filter = { LZMA_FILTER_LZMA1, NULL };
static lzma_filter lzma2_filter = { LZMA_FILTER_LZMA2, NULL };
static lzma_filter delta_filter = { LZMA_FILTER_DELTA, NULL };

static lzma_filter bcj_filters_encoders[] = {
#ifdef HAVE_ENCODER_X86
	{ LZMA_FILTER_X86, NULL },
#endif
#ifdef HAVE_ENCODER_POWERPC
	{ LZMA_FILTER_POWERPC, NULL },
#endif
#ifdef HAVE_ENCODER_IA64
	{ LZMA_FILTER_IA64, NULL },
#endif
#ifdef HAVE_ENCODER_ARM
	{ LZMA_FILTER_ARM, NULL },
#endif
#ifdef HAVE_ENCODER_ARM64
	{ LZMA_FILTER_ARM64, NULL },
#endif
#ifdef HAVE_ENCODER_ARMTHUMB
	{ LZMA_FILTER_ARMTHUMB, NULL },
#endif
#ifdef HAVE_ENCODER_SPARC
	{ LZMA_FILTER_SPARC, NULL },
#endif
};

// HAVE_ENCODERS ifdef not terminated here because decoders are
// only used if encoders are, but encoders can still be used
// even if decoders are not.

#ifdef HAVE_DECODERS
static lzma_filter bcj_filters_decoders[] = {
#ifdef HAVE_DECODER_X86
	{ LZMA_FILTER_X86, NULL },
#endif
#ifdef HAVE_DECODER_POWERPC
	{ LZMA_FILTER_POWERPC, NULL },
#endif
#ifdef HAVE_DECODER_IA64
	{ LZMA_FILTER_IA64, NULL },
#endif
#ifdef HAVE_DECODER_ARM
	{ LZMA_FILTER_ARM, NULL },
#endif
#ifdef HAVE_DECODER_ARM64
	{ LZMA_FILTER_ARM64, NULL },
#endif
#ifdef HAVE_DECODER_ARMTHUMB
	{ LZMA_FILTER_ARMTHUMB, NULL },
#endif
#ifdef HAVE_DECODER_SPARC
	{ LZMA_FILTER_SPARC, NULL },
#endif
};
#endif
#endif


static void
test_lzma_filter_flags_size(void)
{
#ifndef HAVE_ENCODERS
	assert_skip("Encoder support disabled");
#else
	// For each supported filter, test that the size can be calculated
	// and that the size calculated is reasonable. A reasonable size
	// must be greater than 0, but less than the maximum size for the
	// block header.
	uint32_t size = 0;
	if (lzma_filter_encoder_is_supported(LZMA_FILTER_LZMA1)) {
		assert_lzma_ret(lzma_filter_flags_size(&size,
				&lzma1_filter), LZMA_PROG_ERROR);
	}

	if (lzma_filter_encoder_is_supported(LZMA_FILTER_LZMA2)) {
		assert_lzma_ret(lzma_filter_flags_size(&size,
				&lzma2_filter), LZMA_OK);
		assert_true(size != 0 && size < LZMA_BLOCK_HEADER_SIZE_MAX);
	}

	// Do not use macro ARRAY_SIZE() in the for loop condition directly.
	// If the BCJ filters are not configured and built, then ARRAY_SIZE()
	// will return 0 and cause a warning because the for loop will never
	// execute since any unsigned number cannot be < 0 (-Werror=type-limits).
	const uint32_t bcj_array_size = ARRAY_SIZE(bcj_filters_encoders);
	for (uint32_t i = 0; i < bcj_array_size; i++) {
		assert_lzma_ret(lzma_filter_flags_size(&size,
				&bcj_filters_encoders[i]), LZMA_OK);
		assert_true(size != 0 && size < LZMA_BLOCK_HEADER_SIZE_MAX);
	}

	if (lzma_filter_encoder_is_supported(LZMA_FILTER_DELTA)) {
		assert_lzma_ret(lzma_filter_flags_size(&size,
				&delta_filter), LZMA_OK);
		assert_true(size != 0 && size < LZMA_BLOCK_HEADER_SIZE_MAX);
	}

	// Test invalid Filter IDs
	lzma_filter bad_filter = { 2, NULL };

	assert_lzma_ret(lzma_filter_flags_size(&size, &bad_filter),
			LZMA_OPTIONS_ERROR);
	bad_filter.id = LZMA_VLI_MAX;
	assert_lzma_ret(lzma_filter_flags_size(&size, &bad_filter),
			LZMA_PROG_ERROR);
	bad_filter.id = LZMA_FILTER_RESERVED_START;
	assert_lzma_ret(lzma_filter_flags_size(&size, &bad_filter),
			LZMA_PROG_ERROR);
#endif
}


// Helper function for test_lzma_filter_flags_encode.
// The should_encode parameter represents if the encoding operation
// is expected to fail.
// Avoid data -> encode -> decode -> compare to data.
// Instead create expected encoding and compare to result from
// lzma_filter_flags_encode.
// Filter Flags in .xz are encoded as:
// |Filter ID (VLI)|Size of Properties (VLI)|Filter Properties|
#if defined(HAVE_ENCODERS) && defined(HAVE_DECODERS)
static void
verify_filter_flags_encode(lzma_filter *filter, bool should_encode)
{
	uint32_t size = 0;

	// First calculate the size of Filter Flags to know how much
	// memory to allocate to hold the encoded Filter Flags
	assert_lzma_ret(lzma_filter_flags_size(&size, filter), LZMA_OK);
	uint8_t *encoded_out = tuktest_malloc(size * sizeof(uint8_t));
	size_t out_pos = 0;
	if (!should_encode) {
		assert_false(lzma_filter_flags_encode(filter, encoded_out,
				&out_pos, size) == LZMA_OK);
		return;
	}

	// Next encode the Filter Flags for the provided filter
	assert_lzma_ret(lzma_filter_flags_encode(filter, encoded_out,
			&out_pos, size), LZMA_OK);
	assert_uint_eq(size, out_pos);

	// Next decode the VLI for the Filter ID and verify it matches
	// the expected Filter ID
	size_t filter_id_vli_size = 0;
	lzma_vli filter_id = 0;
	assert_lzma_ret(lzma_vli_decode(&filter_id, NULL, encoded_out,
			&filter_id_vli_size, size), LZMA_OK);
	assert_uint_eq(filter->id, filter_id);

	// Next decode the Size of Properties and ensure it equals
	// the expected size.
	// Expected size should be:
	// total filter flag length - size of filter id VLI + size of
	//                            property size VLI
	// Not verifying the contents of Filter Properties since
	// that belongs in a different test
	size_t size_of_properties_vli_size = 0;
	lzma_vli size_of_properties = 0;
	assert_lzma_ret(lzma_vli_decode(&size_of_properties, NULL,
			encoded_out + filter_id_vli_size,
			&size_of_properties_vli_size, size), LZMA_OK);
	assert_uint_eq(size - (size_of_properties_vli_size +
			filter_id_vli_size), size_of_properties);
}
#endif


static void
test_lzma_filter_flags_encode(void)
{
#if !defined(HAVE_ENCODERS) || !defined(HAVE_DECODERS)
	assert_skip("Encoder or decoder support disabled");
#else
	// No test for LZMA1 since the .xz format does not support LZMA1
	// and so the flags cannot be encoded for that filter
	if (lzma_filter_encoder_is_supported(LZMA_FILTER_LZMA2)) {
		// Test with NULL options that should fail
		lzma_options_lzma *options = lzma2_filter.options;
		lzma2_filter.options = NULL;
		verify_filter_flags_encode(&lzma2_filter, false);

		// Place options back in the filter, and test should pass
		lzma2_filter.options = options;
		verify_filter_flags_encode(&lzma2_filter, true);
	}

	// NOTE: Many BCJ filters require that start_offset is a multiple
	// of some power of two. The Filter Flags encoder and decoder don't
	// completely validate the options and thus 257 passes the tests
	// with all BCJ filters. It would be caught when initializing
	// a filter chain encoder or decoder.
	lzma_options_bcj bcj_options = {
		.start_offset = 257
	};

	const uint32_t bcj_array_size = ARRAY_SIZE(bcj_filters_encoders);
	for (uint32_t i = 0; i < bcj_array_size; i++) {
		// NULL options should pass for bcj filters
		verify_filter_flags_encode(&bcj_filters_encoders[i], true);
		lzma_filter bcj_with_options = {
				bcj_filters_encoders[i].id, &bcj_options };
		verify_filter_flags_encode(&bcj_with_options, true);
	}

	if (lzma_filter_encoder_is_supported(LZMA_FILTER_DELTA)) {
		lzma_options_delta delta_opts_below_min = {
			.type = LZMA_DELTA_TYPE_BYTE,
			.dist = LZMA_DELTA_DIST_MIN - 1
		};

		lzma_options_delta delta_opts_above_max = {
			.type = LZMA_DELTA_TYPE_BYTE,
			.dist = LZMA_DELTA_DIST_MAX + 1
		};

		verify_filter_flags_encode(&delta_filter, true);

		lzma_filter delta_filter_bad_options = {
				LZMA_FILTER_DELTA, &delta_opts_below_min };

		// Next test error case using minimum - 1 delta distance
		verify_filter_flags_encode(&delta_filter_bad_options, false);

		// Next test error case using maximum + 1 delta distance
		delta_filter_bad_options.options = &delta_opts_above_max;
		verify_filter_flags_encode(&delta_filter_bad_options, false);

		// Next test NULL case
		delta_filter_bad_options.options = NULL;
		verify_filter_flags_encode(&delta_filter_bad_options, false);
	}

	// Test expected failing cases
	lzma_filter bad_filter = { LZMA_FILTER_RESERVED_START, NULL };
	size_t out_pos = 0;
	size_t out_size = LZMA_BLOCK_HEADER_SIZE_MAX;
	uint8_t out[LZMA_BLOCK_HEADER_SIZE_MAX];


	// Filter ID outside of valid range
	assert_lzma_ret(lzma_filter_flags_encode(&bad_filter, out, &out_pos,
			out_size), LZMA_PROG_ERROR);
	out_pos = 0;
	bad_filter.id = LZMA_VLI_MAX + 1;
	assert_lzma_ret(lzma_filter_flags_encode(&bad_filter, out, &out_pos,
			out_size), LZMA_PROG_ERROR);
	out_pos = 0;

	// Invalid Filter ID
	bad_filter.id = 2;
	assert_lzma_ret(lzma_filter_flags_encode(&bad_filter, out, &out_pos,
			out_size), LZMA_OPTIONS_ERROR);
	out_pos = 0;

	// Out size too small
	if (lzma_filter_encoder_is_supported(LZMA_FILTER_LZMA2)) {
		uint32_t bad_size = 0;

		// First test with 0 output size
		assert_lzma_ret(lzma_filter_flags_encode(
				&lzma2_filter, out, &out_pos, 0),
				LZMA_PROG_ERROR);

		// Next calculate the size needed to encode and
		// use less than that
		assert_lzma_ret(lzma_filter_flags_size(&bad_size,
				&lzma2_filter), LZMA_OK);

		assert_lzma_ret(lzma_filter_flags_encode(
				&lzma2_filter, out, &out_pos,
				bad_size - 1), LZMA_PROG_ERROR);
		out_pos = 0;
	}

	// Invalid options
	if (lzma_filter_encoder_is_supported(LZMA_FILTER_DELTA)) {
		bad_filter.id = LZMA_FILTER_DELTA;

		// First test with NULL options
		assert_lzma_ret(lzma_filter_flags_encode(&bad_filter, out,
				&out_pos, out_size), LZMA_PROG_ERROR);
		out_pos = 0;

		// Next test with invalid options
		lzma_options_delta bad_options = {
			.dist = LZMA_DELTA_DIST_MAX + 1,
			.type = LZMA_DELTA_TYPE_BYTE
		};
		bad_filter.options = &bad_options;

		assert_lzma_ret(lzma_filter_flags_encode(&bad_filter, out,
				&out_pos, out_size), LZMA_PROG_ERROR);
	}
#endif
}


// Helper function for test_lzma_filter_flags_decode.
// Encodes the filter_in without using lzma_filter_flags_encode.
// Leaves the specific assertions of filter_out options to the caller
// because it is agnostic to the type of options used in the call
#if defined(HAVE_ENCODERS) && defined(HAVE_DECODERS)
static void
verify_filter_flags_decode(lzma_filter *filter_in, lzma_filter *filter_out)
{
	uint32_t total_size = 0;

	assert_lzma_ret(lzma_filter_flags_size(&total_size, filter_in),
			LZMA_OK);
	assert_uint(total_size, >, 0);
	uint8_t *filter_flag_buffer = tuktest_malloc(total_size);

	uint32_t properties_size = 0;
	size_t out_pos = 0;
	size_t in_pos = 0;
	assert_lzma_ret(lzma_properties_size(&properties_size, filter_in),
			LZMA_OK);
	assert_lzma_ret(lzma_vli_encode(filter_in->id, NULL,
			filter_flag_buffer, &out_pos, total_size), LZMA_OK);
	assert_lzma_ret(lzma_vli_encode(properties_size, NULL,
			filter_flag_buffer, &out_pos, total_size),
			LZMA_OK);
	assert_lzma_ret(lzma_properties_encode(filter_in,
			filter_flag_buffer + out_pos), LZMA_OK);
	assert_lzma_ret(lzma_filter_flags_decode(filter_out, NULL,
			filter_flag_buffer, &in_pos, total_size),
			LZMA_OK);
	assert_uint_eq(filter_in->id, filter_out->id);
}
#endif


static void
test_lzma_filter_flags_decode(void)
{
#if !defined(HAVE_ENCODERS) || !defined(HAVE_DECODERS)
	assert_skip("Encoder or decoder support disabled");
#else
	// For each filter, only run the decoder test if both the encoder
	// and decoder are enabled. This is because verify_filter_flags_decode
	// uses lzma_filter_flags_size which requires the encoder.
	if (lzma_filter_decoder_is_supported(LZMA_FILTER_LZMA2) &&
			lzma_filter_encoder_is_supported(LZMA_FILTER_LZMA2)) {
		lzma_filter lzma2_decoded = { LZMA_FILTER_LZMA2, NULL };

		verify_filter_flags_decode(&lzma2_filter, &lzma2_decoded);

		lzma_options_lzma *expected = lzma2_filter.options;
		lzma_options_lzma *decoded = lzma2_decoded.options;

		// Only the dictionary size is encoded and decoded
		// so only compare those
		assert_uint_eq(decoded->dict_size, expected->dict_size);

		// The decoded options must be freed by the caller
		free(decoded);
	}

	const uint32_t bcj_array_size = ARRAY_SIZE(bcj_filters_decoders);
	for (uint32_t i = 0; i < bcj_array_size; i++) {
		if (lzma_filter_encoder_is_supported(
				bcj_filters_decoders[i].id)) {
			lzma_filter bcj_decoded = {
					bcj_filters_decoders[i].id, NULL };

			lzma_filter bcj_encoded = {
					bcj_filters_decoders[i].id, NULL };

			// First test without options
			verify_filter_flags_decode(&bcj_encoded,
					&bcj_decoded);
			assert_true(bcj_decoded.options == NULL);

			// Next test with offset
			lzma_options_bcj options = {
				.start_offset = 257
			};

			bcj_encoded.options = &options;
			verify_filter_flags_decode(&bcj_encoded,
					&bcj_decoded);
			lzma_options_bcj *decoded_opts = bcj_decoded.options;
			assert_uint_eq(decoded_opts->start_offset,
					options.start_offset);
			free(decoded_opts);
		}
	}

	if (lzma_filter_decoder_is_supported(LZMA_FILTER_DELTA) &&
			lzma_filter_encoder_is_supported(LZMA_FILTER_DELTA)) {
		lzma_filter delta_decoded = { LZMA_FILTER_DELTA, NULL };

		verify_filter_flags_decode(&delta_filter, &delta_decoded);
		lzma_options_delta *expected = delta_filter.options;
		lzma_options_delta *decoded = delta_decoded.options;
		assert_uint_eq(expected->dist, decoded->dist);
		assert_uint_eq(expected->type, decoded->type);

		free(decoded);
	}

	// Test expected failing cases
	uint8_t bad_encoded_filter[LZMA_BLOCK_HEADER_SIZE_MAX];
	lzma_filter bad_filter;

	// Filter ID outside of valid range
	lzma_vli bad_filter_id = LZMA_FILTER_RESERVED_START;
	size_t bad_encoded_out_pos = 0;
	size_t in_pos = 0;

	assert_lzma_ret(lzma_vli_encode(bad_filter_id, NULL,
			bad_encoded_filter, &bad_encoded_out_pos,
			LZMA_BLOCK_HEADER_SIZE_MAX), LZMA_OK);

	assert_lzma_ret(lzma_filter_flags_decode(&bad_filter, NULL,
			bad_encoded_filter, &in_pos,
			LZMA_BLOCK_HEADER_SIZE_MAX), LZMA_DATA_ERROR);

	bad_encoded_out_pos = 0;
	in_pos = 0;

	// Invalid Filter ID
	bad_filter_id = 2;
	bad_encoded_out_pos = 0;
	in_pos = 0;

	assert_lzma_ret(lzma_vli_encode(bad_filter_id, NULL,
			bad_encoded_filter, &bad_encoded_out_pos,
			LZMA_BLOCK_HEADER_SIZE_MAX), LZMA_OK);

	// Next encode Size of Properties with the value of 0
	assert_lzma_ret(lzma_vli_encode(0, NULL,
			bad_encoded_filter, &bad_encoded_out_pos,
			LZMA_BLOCK_HEADER_SIZE_MAX), LZMA_OK);

	// Decode should fail on bad Filter ID
	assert_lzma_ret(lzma_filter_flags_decode(&bad_filter, NULL,
			bad_encoded_filter, &in_pos,
			LZMA_BLOCK_HEADER_SIZE_MAX), LZMA_OPTIONS_ERROR);
	bad_encoded_out_pos = 0;
	in_pos = 0;

	// Outsize too small
	// Encode the LZMA2 filter normally, but then set
	// the out size when decoding as too small
	if (lzma_filter_encoder_is_supported(LZMA_FILTER_LZMA2) &&
			lzma_filter_decoder_is_supported(LZMA_FILTER_LZMA2)) {
		uint32_t filter_flag_size = 0;
		assert_lzma_ret(lzma_filter_flags_size(&filter_flag_size,
				&lzma2_filter), LZMA_OK);

		assert_lzma_ret(lzma_filter_flags_encode(&lzma2_filter,
				bad_encoded_filter, &bad_encoded_out_pos,
				LZMA_BLOCK_HEADER_SIZE_MAX), LZMA_OK);

		assert_lzma_ret(lzma_filter_flags_decode(&bad_filter, NULL,
				bad_encoded_filter, &in_pos,
				filter_flag_size - 1), LZMA_DATA_ERROR);
	}
#endif
}


extern int
main(int argc, char **argv)
{
	tuktest_start(argc, argv);

#ifdef HAVE_ENCODERS
	// Only init filter options if encoder is supported because decoder
	// tests requires encoder support, so the decoder tests will only
	// run if for a given filter both the encoder and decoder are enabled.
	if (lzma_filter_encoder_is_supported(LZMA_FILTER_LZMA1)) {
		lzma_options_lzma *options = tuktest_malloc(
				sizeof(lzma_options_lzma));
		lzma_lzma_preset(options, LZMA_PRESET_DEFAULT);
		lzma1_filter.options = options;
	}

	if (lzma_filter_encoder_is_supported(LZMA_FILTER_LZMA2)) {
		lzma_options_lzma *options = tuktest_malloc(
				sizeof(lzma_options_lzma));
		lzma_lzma_preset(options, LZMA_PRESET_DEFAULT);
		lzma2_filter.options = options;
	}

	if (lzma_filter_encoder_is_supported(LZMA_FILTER_DELTA)) {
		lzma_options_delta *options = tuktest_malloc(
				sizeof(lzma_options_delta));
		options->dist = LZMA_DELTA_DIST_MIN;
		options->type = LZMA_DELTA_TYPE_BYTE;
		delta_filter.options = options;
	}
#endif

	tuktest_run(test_lzma_filter_flags_size);
	tuktest_run(test_lzma_filter_flags_encode);
	tuktest_run(test_lzma_filter_flags_decode);
	return tuktest_end();
}
