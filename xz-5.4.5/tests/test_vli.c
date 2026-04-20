///////////////////////////////////////////////////////////////////////////////
//
/// \file       test_vli.c
/// \brief      Tests liblzma vli functions
//
//  Author:     Jia Tan
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "tests.h"


// Pre-encoded VLI values for testing
// VLI can have between 1 and 9 bytes when encoded
// They are encoded little endian where all but the last
// byte must have the leading 1 bit set
#if defined(HAVE_ENCODERS) || defined(HAVE_DECODERS)
static const uint8_t one_byte[1] = {0x25};
static const lzma_vli one_byte_value = 37;

static const uint8_t two_bytes[2] = {0x80, 0x56};
static const lzma_vli two_byte_value = 11008;

static const uint8_t three_bytes[3] = {0x99, 0x92, 0x20};
static const lzma_vli three_byte_value = 526617;

static const uint8_t four_bytes[4] = {0x97, 0x83, 0x94, 0x47};
static const lzma_vli four_byte_value = 149225879;

static const uint8_t five_bytes[5] = {0xA6, 0x92, 0x88, 0x89, 0x32};
static const lzma_vli five_byte_value = 13440780582;

static const uint8_t six_bytes[6] = {0xA9, 0x84, 0x99, 0x82, 0x94, 0x12};
static const lzma_vli six_byte_value = 623848604201;

static const uint8_t seven_bytes[7] = {0x90, 0x80, 0x90, 0x80, 0x90, 0x80,
				0x79};
static const lzma_vli seven_byte_value = 532167923073040;

static const uint8_t eight_bytes[8] = {0x91, 0x87, 0xF2, 0xB2, 0xC2, 0xD2,
				0x93, 0x63};
static const lzma_vli eight_byte_value = 55818443594433425;

static const uint8_t nine_bytes[9] = {0x81, 0x91, 0xA1, 0xB1, 0xC1, 0xD1,
				0xE1, 0xF1, 0x1};
static const lzma_vli nine_byte_value = 136100349976529025;
#endif


static void
test_lzma_vli_size(void)
{
	// First test invalid VLI values (should return 0)
	// VLI UNKNOWN is an invalid VLI
	assert_uint_eq(lzma_vli_size(LZMA_VLI_UNKNOWN), 0);
	// Loop over a few VLI values just over the maximum
	for (uint64_t i = LZMA_VLI_MAX + 1; i < LZMA_VLI_MAX + 10; i++)
		assert_uint_eq(lzma_vli_size(i), 0);

	// Number should increment every seven set bits
	lzma_vli vli = 1;
	for (uint32_t i = 1; i < LZMA_VLI_BYTES_MAX; i++, vli <<= 7) {
		// Test the base value and a few others around it
		assert_uint_eq(lzma_vli_size(vli), i);
		assert_uint_eq(lzma_vli_size(vli * 2), i);
		assert_uint_eq(lzma_vli_size(vli + 10), i);
		assert_uint_eq(lzma_vli_size(vli * 3 + 39), i);
	}
}


#ifdef HAVE_ENCODERS
// Helper function for test_lzma_vli_encode
// Encodes an input VLI and compares against a pre-computed value
static void
encode_single_call_mode(lzma_vli input, const uint8_t *expected,
		uint32_t expected_len)
{
	uint8_t out[LZMA_VLI_BYTES_MAX];
	size_t out_pos = 0;
	assert_lzma_ret(lzma_vli_encode(input, NULL, out, &out_pos,
			expected_len), LZMA_OK);
	assert_uint_eq(out_pos, expected_len);
	assert_array_eq(out, expected, expected_len);
}


// Helper function for test_lzma_vli_encode
// Encodes an input VLI one byte at a time with the multi call
// method. Then compares against a pre-computed value
static void
encode_multi_call_mode(lzma_vli input, const uint8_t *expected,
		uint32_t expected_len)
{
	uint8_t out[LZMA_VLI_BYTES_MAX];
	size_t out_pos = 0;
	size_t vli_pos = 0;

	for (uint32_t i = 1; i < expected_len; i++) {
		assert_lzma_ret(lzma_vli_encode(input, &vli_pos, out,
				&out_pos, i), LZMA_OK);
		assert_uint_eq(out_pos, i);
		assert_uint_eq(vli_pos, i);
	}
	assert_lzma_ret(lzma_vli_encode(input, &vli_pos, out, &out_pos,
			expected_len), LZMA_STREAM_END);
	assert_uint_eq(out_pos, expected_len);
	assert_uint_eq(vli_pos, expected_len);
	assert_array_eq(out, expected, expected_len);
}
#endif


static void
test_lzma_vli_encode(void)
{
#ifndef HAVE_ENCODERS
	assert_skip("Encoder support disabled");
#else
	size_t vli_pos = 0;
	uint8_t out[LZMA_VLI_BYTES_MAX];
	uint8_t zeros[LZMA_VLI_BYTES_MAX];
	memzero(out, LZMA_VLI_BYTES_MAX);
	memzero(zeros, LZMA_VLI_BYTES_MAX);
	size_t out_pos = 0;

	// First test invalid input parameters
	// VLI invalid
	assert_lzma_ret(lzma_vli_encode(LZMA_VLI_UNKNOWN, &vli_pos, out,
			&out_pos, sizeof(out)), LZMA_PROG_ERROR);
	// Failure should not change params
	assert_uint_eq(vli_pos, 0);
	assert_uint_eq(out_pos, 0);
	assert_array_eq(out, zeros, LZMA_VLI_BYTES_MAX);

	assert_lzma_ret(lzma_vli_encode(LZMA_VLI_MAX + 1, &vli_pos, out,
		&out_pos, sizeof(out)), LZMA_PROG_ERROR);
	assert_uint_eq(vli_pos, 0);
	assert_uint_eq(out_pos, 0);
	assert_array_eq(out, zeros, LZMA_VLI_BYTES_MAX);

	// 0 output size
	assert_lzma_ret(lzma_vli_encode(one_byte_value, &vli_pos, out,
			&out_pos, 0), LZMA_BUF_ERROR);
	assert_uint_eq(vli_pos, 0);
	assert_uint_eq(out_pos, 0);
	assert_array_eq(out, zeros, LZMA_VLI_BYTES_MAX);

	// Size of VLI does not fit in buffer
	size_t phony_out_pos = 3;
	assert_lzma_ret(lzma_vli_encode(one_byte_value, NULL, out,
			&phony_out_pos, 2), LZMA_PROG_ERROR);

	assert_lzma_ret(lzma_vli_encode(LZMA_VLI_MAX / 2, NULL, out,
			&out_pos, 2), LZMA_PROG_ERROR);

	// Test single-call mode (using vli_pos as NULL)
	encode_single_call_mode(one_byte_value, one_byte,
			sizeof(one_byte));
	encode_single_call_mode(two_byte_value, two_bytes,
			sizeof(two_bytes));
	encode_single_call_mode(three_byte_value, three_bytes,
			sizeof(three_bytes));
	encode_single_call_mode(four_byte_value, four_bytes,
			sizeof(four_bytes));
	encode_single_call_mode(five_byte_value, five_bytes,
			sizeof(five_bytes));
	encode_single_call_mode(six_byte_value, six_bytes,
			sizeof(six_bytes));
	encode_single_call_mode(seven_byte_value, seven_bytes,
			sizeof(seven_bytes));
	encode_single_call_mode(eight_byte_value, eight_bytes,
			sizeof(eight_bytes));
	encode_single_call_mode(nine_byte_value, nine_bytes,
			sizeof(nine_bytes));

	// Test multi-call mode
	encode_multi_call_mode(one_byte_value, one_byte,
			sizeof(one_byte));
	encode_multi_call_mode(two_byte_value, two_bytes,
			sizeof(two_bytes));
	encode_multi_call_mode(three_byte_value, three_bytes,
			sizeof(three_bytes));
	encode_multi_call_mode(four_byte_value, four_bytes,
			sizeof(four_bytes));
	encode_multi_call_mode(five_byte_value, five_bytes,
			sizeof(five_bytes));
	encode_multi_call_mode(six_byte_value, six_bytes,
			sizeof(six_bytes));
	encode_multi_call_mode(seven_byte_value, seven_bytes,
			sizeof(seven_bytes));
	encode_multi_call_mode(eight_byte_value, eight_bytes,
			sizeof(eight_bytes));
	encode_multi_call_mode(nine_byte_value, nine_bytes,
			sizeof(nine_bytes));
#endif
}


#ifdef HAVE_DECODERS
static void
decode_single_call_mode(const uint8_t *input, uint32_t input_len,
		lzma_vli expected)
{
	lzma_vli out = 0;
	size_t in_pos = 0;

	assert_lzma_ret(lzma_vli_decode(&out, NULL, input, &in_pos,
			input_len), LZMA_OK);
	assert_uint_eq(in_pos, input_len);
	assert_uint_eq(out, expected);
}


static void
decode_multi_call_mode(const uint8_t *input, uint32_t input_len,
		lzma_vli expected)
{
	lzma_vli out = 0;
	size_t in_pos = 0;
	size_t vli_pos = 0;

	for (uint32_t i = 1; i < input_len; i++) {
		assert_lzma_ret(lzma_vli_decode(&out, &vli_pos, input,
				&in_pos, i), LZMA_OK);
		assert_uint_eq(in_pos, i);
		assert_uint_eq(vli_pos, i);
	}

	assert_lzma_ret(lzma_vli_decode(&out, &vli_pos, input, &in_pos,
			input_len), LZMA_STREAM_END);
	assert_uint_eq(in_pos, input_len);
	assert_uint_eq(vli_pos, input_len);
	assert_uint_eq(out, expected);
}
#endif


static void
test_lzma_vli_decode(void)
{
#ifndef HAVE_DECODERS
	assert_skip("Decoder support disabled");
#else
	lzma_vli out = 0;
	size_t in_pos = 0;

	// First test invalid input params
	// 0 in_size
	assert_lzma_ret(lzma_vli_decode(&out, NULL, one_byte, &in_pos, 0),
			LZMA_DATA_ERROR);
	assert_uint_eq(out, 0);
	assert_uint_eq(in_pos, 0);

	// VLI encoded is invalid (last digit has leading 1 set)
	uint8_t invalid_vli[3] = {0x80, 0x80, 0x80};
	assert_lzma_ret(lzma_vli_decode(&out, NULL, invalid_vli, &in_pos,
			sizeof(invalid_vli)), LZMA_DATA_ERROR);

	// Bad vli_pos
	size_t vli_pos = LZMA_VLI_BYTES_MAX;
	assert_lzma_ret(lzma_vli_decode(&out, &vli_pos, invalid_vli, &in_pos,
			sizeof(invalid_vli)), LZMA_PROG_ERROR);

	// Bad in_pos
	in_pos = sizeof(invalid_vli);
	assert_lzma_ret(lzma_vli_decode(&out, &in_pos, invalid_vli, &in_pos,
			sizeof(invalid_vli)), LZMA_BUF_ERROR);

	// Test single call mode
	decode_single_call_mode(one_byte, sizeof(one_byte),
			one_byte_value);
	decode_single_call_mode(two_bytes, sizeof(two_bytes),
			two_byte_value);
	decode_single_call_mode(three_bytes, sizeof(three_bytes),
			three_byte_value);
	decode_single_call_mode(four_bytes, sizeof(four_bytes),
			four_byte_value);
	decode_single_call_mode(five_bytes, sizeof(five_bytes),
			five_byte_value);
	decode_single_call_mode(six_bytes, sizeof(six_bytes),
			six_byte_value);
	decode_single_call_mode(seven_bytes, sizeof(seven_bytes),
			seven_byte_value);
	decode_single_call_mode(eight_bytes, sizeof(eight_bytes),
			eight_byte_value);
	decode_single_call_mode(nine_bytes, sizeof(nine_bytes),
			nine_byte_value);

	// Test multi call mode
	decode_multi_call_mode(one_byte, sizeof(one_byte),
			one_byte_value);
	decode_multi_call_mode(two_bytes, sizeof(two_bytes),
			two_byte_value);
	decode_multi_call_mode(three_bytes, sizeof(three_bytes),
			three_byte_value);
	decode_multi_call_mode(four_bytes, sizeof(four_bytes),
			four_byte_value);
	decode_multi_call_mode(five_bytes, sizeof(five_bytes),
			five_byte_value);
	decode_multi_call_mode(six_bytes, sizeof(six_bytes),
			six_byte_value);
	decode_multi_call_mode(seven_bytes, sizeof(seven_bytes),
			seven_byte_value);
	decode_multi_call_mode(eight_bytes, sizeof(eight_bytes),
			eight_byte_value);
	decode_multi_call_mode(nine_bytes, sizeof(nine_bytes),
			nine_byte_value);
#endif
}


extern int
main(int argc, char **argv)
{
	tuktest_start(argc, argv);
	tuktest_run(test_lzma_vli_size);
	tuktest_run(test_lzma_vli_encode);
	tuktest_run(test_lzma_vli_decode);
	return tuktest_end();
}
