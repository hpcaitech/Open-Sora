///////////////////////////////////////////////////////////////////////////////
//
/// \file       test_stream_flags.c
/// \brief      Tests Stream Header and Stream Footer coders
//
//  Authors:    Jia Tan
//              Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "tests.h"


// Size of the Stream Flags field
// (taken from src/liblzma/common/stream_flags_common.h)
#define XZ_STREAM_FLAGS_SIZE 2

#ifdef HAVE_ENCODERS
// Header and footer magic bytes for .xz file format
// (taken from src/liblzma/common/stream_flags_common.c)
static const uint8_t xz_header_magic[6]
		= { 0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00 };
static const uint8_t xz_footer_magic[2] = { 0x59, 0x5A };
#endif


#ifdef HAVE_ENCODERS
static void
stream_header_encode_helper(lzma_check check)
{
	lzma_stream_flags flags = {
		.version = 0,
		.check = check,
	};

	uint8_t header[LZMA_STREAM_HEADER_SIZE];

	// Encode Stream Header
	assert_lzma_ret(lzma_stream_header_encode(&flags, header), LZMA_OK);

	// Stream Header must start with Header Magic Bytes
	const uint32_t magic_size = sizeof(xz_header_magic);
	assert_array_eq(header, xz_header_magic, magic_size);

	// Next must come Stream Flags
	const uint8_t *encoded_stream_flags = header + magic_size;

	// First byte is always null-byte.
	// Second byte must have the Check ID in the lowest four bits
	// and the highest four bits zero.
	const uint8_t expected_stream_flags[] = { 0, check };
	assert_array_eq(encoded_stream_flags, expected_stream_flags,
			XZ_STREAM_FLAGS_SIZE);

	// Last part is the CRC32 of the Stream Flags
	const uint8_t *crc_ptr = encoded_stream_flags + XZ_STREAM_FLAGS_SIZE;
	const uint32_t expected_crc = lzma_crc32(expected_stream_flags,
			XZ_STREAM_FLAGS_SIZE, 0);
	assert_uint_eq(read32le(crc_ptr), expected_crc);
}
#endif


static void
test_lzma_stream_header_encode(void)
{
#ifndef HAVE_ENCODERS
	assert_skip("Encoder support disabled");
#else
	for (lzma_check i = 0; i < LZMA_CHECK_ID_MAX; i++)
		stream_header_encode_helper(i);

	lzma_stream_flags flags = {
		.version = 0,
		.check = LZMA_CHECK_CRC32
	};

	uint8_t header[LZMA_STREAM_HEADER_SIZE];

	// Should fail if version > 0
	flags.version = 1;
	assert_lzma_ret(lzma_stream_header_encode(&flags, header),
			LZMA_OPTIONS_ERROR);
	flags.version = 0;

	// Should fail if Check ID is invalid
	flags.check = INVALID_LZMA_CHECK_ID;
	assert_lzma_ret(lzma_stream_header_encode(&flags, header),
			LZMA_PROG_ERROR);
	flags.check = LZMA_CHECK_CRC32;

	// Should pass even if Backward Size is invalid
	// because Stream Header doesn't have that field.
	flags.backward_size = LZMA_VLI_MAX + 1;
	assert_lzma_ret(lzma_stream_header_encode(&flags, header), LZMA_OK);
#endif
}


#if defined(HAVE_ENCODERS)
static void
stream_footer_encode_helper(lzma_check check)
{
	lzma_stream_flags flags = {
		.version = 0,
		.check = check,
		.backward_size = LZMA_BACKWARD_SIZE_MIN,
	};

	uint8_t footer[LZMA_STREAM_HEADER_SIZE];

	// Encode Stream Footer
	assert_lzma_ret(lzma_stream_footer_encode(&flags, footer), LZMA_OK);

	// Stream Footer must start with CRC32
	const uint32_t crc = read32le(footer);
	const uint32_t expected_crc = lzma_crc32(footer + sizeof(uint32_t),
			LZMA_STREAM_HEADER_SIZE - (sizeof(uint32_t) +
			sizeof(xz_footer_magic)), 0);
	assert_uint_eq(crc, expected_crc);

	// Next the Backward Size
	const uint32_t backwards_size = read32le(footer + sizeof(uint32_t));
	const uint32_t expected_backwards_size = flags.backward_size / 4 - 1;
	assert_uint_eq(backwards_size, expected_backwards_size);

	// Next the Stream Flags
	const uint8_t *stream_flags = footer + sizeof(uint32_t) * 2;

	// First byte must be null
	assert_uint_eq(stream_flags[0], 0);

	// Second byte must have the Check ID in the lowest four bits
	// and the highest four bits zero.
	assert_uint_eq(stream_flags[1], check);

	// And ends with Footer Magic Bytes
	const uint8_t *expected_footer_magic = stream_flags +
			XZ_STREAM_FLAGS_SIZE;
	assert_array_eq(expected_footer_magic, xz_footer_magic,
			sizeof(xz_footer_magic));
}
#endif


static void
test_lzma_stream_footer_encode(void)
{
#ifndef HAVE_ENCODERS
	assert_skip("Encoder support disabled");
#else
	for (lzma_check i = 0; i < LZMA_CHECK_ID_MAX; i++)
		stream_footer_encode_helper(i);

	lzma_stream_flags flags = {
		.version = 0,
		.backward_size = LZMA_BACKWARD_SIZE_MIN,
		.check = LZMA_CHECK_CRC32
	};

	uint8_t footer[LZMA_STREAM_HEADER_SIZE];

	// Should fail if version > 0
	flags.version = 1;
	assert_lzma_ret(lzma_stream_footer_encode(&flags, footer),
			LZMA_OPTIONS_ERROR);
	flags.version = 0;

	// Should fail if Check ID is invalid
	flags.check = INVALID_LZMA_CHECK_ID;
	assert_lzma_ret(lzma_stream_footer_encode(&flags, footer),
			LZMA_PROG_ERROR);

	// Should fail if Backward Size is invalid
	flags.backward_size -= 1;
	assert_lzma_ret(lzma_stream_footer_encode(&flags, footer),
			LZMA_PROG_ERROR);
	flags.backward_size += 2;
	assert_lzma_ret(lzma_stream_footer_encode(&flags, footer),
			LZMA_PROG_ERROR);
	flags.backward_size = LZMA_BACKWARD_SIZE_MAX + 4;
	assert_lzma_ret(lzma_stream_footer_encode(&flags, footer),
			LZMA_PROG_ERROR);
#endif
}


#if defined(HAVE_ENCODERS) && defined(HAVE_DECODERS)
static void
stream_header_decode_helper(lzma_check check)
{
	lzma_stream_flags flags = {
		.version = 0,
		.check = check
	};

	uint8_t header[LZMA_STREAM_HEADER_SIZE];

	assert_lzma_ret(lzma_stream_header_encode(&flags, header), LZMA_OK);

	lzma_stream_flags dest_flags;
	assert_lzma_ret(lzma_stream_header_decode(&dest_flags, header),
			LZMA_OK);

	// Version should be 0
	assert_uint_eq(dest_flags.version, 0);

	// Backward Size should be LZMA_VLI_UNKNOWN
	assert_uint_eq(dest_flags.backward_size, LZMA_VLI_UNKNOWN);

	// Check ID must equal the argument given to this function.
	assert_uint_eq(dest_flags.check, check);
}
#endif


static void
test_lzma_stream_header_decode(void)
{
#if !defined(HAVE_ENCODERS) || !defined(HAVE_DECODERS)
	assert_skip("Encoder or decoder support disabled");
#else
	for (lzma_check i = 0; i < LZMA_CHECK_ID_MAX; i++)
		stream_header_decode_helper(i);

	lzma_stream_flags flags = {
		.version = 0,
		.check = LZMA_CHECK_CRC32
	};

	uint8_t header[LZMA_STREAM_HEADER_SIZE];
	lzma_stream_flags dest;

	// First encode known flags to header buffer
	assert_lzma_ret(lzma_stream_header_encode(&flags, header), LZMA_OK);

	// Should fail if magic bytes do not match
	header[0] ^= 1;
	assert_lzma_ret(lzma_stream_header_decode(&dest, header),
			LZMA_FORMAT_ERROR);
	header[0] ^= 1;

	// Should fail if a reserved bit is set
	uint8_t *stream_flags = header + sizeof(xz_header_magic);
	stream_flags[0] = 1;

	// Need to adjust CRC32 after making a change since the CRC32
	// is verified before decoding the Stream Flags field.
	uint8_t *crc32_ptr = header + sizeof(xz_header_magic)
			+ XZ_STREAM_FLAGS_SIZE;
	const uint32_t crc_orig = read32le(crc32_ptr);
	uint32_t new_crc32 = lzma_crc32(
			stream_flags, XZ_STREAM_FLAGS_SIZE, 0);
	write32le(crc32_ptr, new_crc32);
	assert_lzma_ret(lzma_stream_header_decode(&dest, header),
			LZMA_OPTIONS_ERROR);
	stream_flags[0] = 0;
	write32le(crc32_ptr, crc_orig);

	// Should fail if upper bits of check ID are set
	stream_flags[1] |= 0xF0;
	new_crc32 = lzma_crc32(stream_flags, XZ_STREAM_FLAGS_SIZE, 0);
	write32le(crc32_ptr, new_crc32);
	assert_lzma_ret(lzma_stream_header_decode(&dest, header),
			LZMA_OPTIONS_ERROR);
	stream_flags[1] = flags.check;
	write32le(crc32_ptr, crc_orig);

	// Should fail if CRC32 does not match.
	// First, alter a byte in the Stream Flags.
	stream_flags[0] = 1;
	assert_lzma_ret(lzma_stream_header_decode(&dest, header),
			LZMA_DATA_ERROR);
	stream_flags[0] = 0;

	// Next, change the CRC32.
	*crc32_ptr ^= 1;
	assert_lzma_ret(lzma_stream_header_decode(&dest, header),
			LZMA_DATA_ERROR);
#endif
}


#if defined(HAVE_ENCODERS) && defined(HAVE_DECODERS)
static void
stream_footer_decode_helper(lzma_check check)
{
	lzma_stream_flags flags = {
		.version = 0,
		.backward_size = LZMA_BACKWARD_SIZE_MIN,
		.check = check,
	};

	uint8_t footer[LZMA_STREAM_HEADER_SIZE];
	assert_lzma_ret(lzma_stream_footer_encode(&flags, footer), LZMA_OK);

	lzma_stream_flags dest_flags;
	assert_lzma_ret(lzma_stream_footer_decode(&dest_flags, footer),
			LZMA_OK);

	// Version should be 0.
	assert_uint_eq(dest_flags.version, 0);

	// Backward Size should equal the value from the flags.
	assert_uint_eq(dest_flags.backward_size, flags.backward_size);

	// Check ID must equal argument given to this function.
	assert_uint_eq(dest_flags.check, check);
}
#endif


static void
test_lzma_stream_footer_decode(void)
{
#if !defined(HAVE_ENCODERS) || !defined(HAVE_DECODERS)
	assert_skip("Encoder or decoder support disabled");
#else
	for (lzma_check i = 0; i < LZMA_CHECK_ID_MAX; i++)
		stream_footer_decode_helper(i);

	lzma_stream_flags flags = {
		.version = 0,
		.check = LZMA_CHECK_CRC32,
		.backward_size = LZMA_BACKWARD_SIZE_MIN
	};

	uint8_t footer[LZMA_STREAM_HEADER_SIZE];
	lzma_stream_flags dest;

	// First encode known flags to the footer buffer
	assert_lzma_ret(lzma_stream_footer_encode(&flags, footer), LZMA_OK);

	// Should fail if magic bytes do not match
	footer[LZMA_STREAM_HEADER_SIZE - 1] ^= 1;
	assert_lzma_ret(lzma_stream_footer_decode(&dest, footer),
			LZMA_FORMAT_ERROR);
	footer[LZMA_STREAM_HEADER_SIZE - 1] ^= 1;

	// Should fail if a reserved bit is set.
	// In the Stream Footer, the Stream Flags follow the CRC32 (4 bytes)
	// and the Backward Size (4 bytes)
	uint8_t *stream_flags = footer + sizeof(uint32_t) * 2;
	stream_flags[0] = 1;

	// Need to adjust the CRC32 so it will not fail that check instead
	uint8_t *crc32_ptr = footer;
	const uint32_t crc_orig = read32le(crc32_ptr);
	uint8_t *backward_size = footer + sizeof(uint32_t);
	uint32_t new_crc32 = lzma_crc32(backward_size, sizeof(uint32_t) +
			XZ_STREAM_FLAGS_SIZE, 0);
	write32le(crc32_ptr, new_crc32);
	assert_lzma_ret(lzma_stream_footer_decode(&dest, footer),
			LZMA_OPTIONS_ERROR);
	stream_flags[0] = 0;
	write32le(crc32_ptr, crc_orig);

	// Should fail if upper bits of check ID are set
	stream_flags[1] |= 0xF0;
	new_crc32 = lzma_crc32(backward_size, sizeof(uint32_t) +
			XZ_STREAM_FLAGS_SIZE, 0);
	write32le(crc32_ptr, new_crc32);
	assert_lzma_ret(lzma_stream_footer_decode(&dest, footer),
			LZMA_OPTIONS_ERROR);
	stream_flags[1] = flags.check;
	write32le(crc32_ptr, crc_orig);

	// Should fail if CRC32 does not match.
	// First, alter a byte in the Stream Flags.
	stream_flags[0] = 1;
	assert_lzma_ret(lzma_stream_footer_decode(&dest, footer),
			LZMA_DATA_ERROR);
	stream_flags[0] = 0;

	// Next, change the CRC32
	*crc32_ptr ^= 1;
	assert_lzma_ret(lzma_stream_footer_decode(&dest, footer),
			LZMA_DATA_ERROR);
#endif
}


static void
test_lzma_stream_flags_compare(void)
{
	lzma_stream_flags first = {
		.version = 0,
		.backward_size = LZMA_BACKWARD_SIZE_MIN,
		.check = LZMA_CHECK_CRC32,
	};

	lzma_stream_flags second = first;

	// First test should pass
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second), LZMA_OK);

	// Altering either version should cause an error
	first.version = 1;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second),
			LZMA_OPTIONS_ERROR);
	second.version = 1;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second),
			LZMA_OPTIONS_ERROR);
	first.version = 0;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second),
			LZMA_OPTIONS_ERROR);
	second.version = 0;

	// Check types must be under the maximum
	first.check = INVALID_LZMA_CHECK_ID;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second),
			LZMA_PROG_ERROR);
	second.check = INVALID_LZMA_CHECK_ID;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second),
			LZMA_PROG_ERROR);
	first.check = LZMA_CHECK_CRC32;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second),
			LZMA_PROG_ERROR);
	second.check = LZMA_CHECK_CRC32;

	// Check types must be equal
	for (lzma_check i = 0; i < LZMA_CHECK_ID_MAX; i++) {
		first.check = i;
		if (i == second.check)
			assert_lzma_ret(lzma_stream_flags_compare(&first,
					&second), LZMA_OK);
		else
			assert_lzma_ret(lzma_stream_flags_compare(&first,
					&second), LZMA_DATA_ERROR);
	}
	first.check = LZMA_CHECK_CRC32;

	// Backward Size comparison is skipped if either are LZMA_VLI_UNKNOWN
	first.backward_size = LZMA_VLI_UNKNOWN;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second), LZMA_OK);
	second.backward_size = LZMA_VLI_MAX + 1;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second), LZMA_OK);
	second.backward_size = LZMA_BACKWARD_SIZE_MIN;

	// Backward Sizes need to be valid
	first.backward_size = LZMA_VLI_MAX + 4;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second),
			LZMA_PROG_ERROR);
	second.backward_size = LZMA_VLI_MAX + 4;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second),
			LZMA_PROG_ERROR);
	first.backward_size = LZMA_BACKWARD_SIZE_MIN;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second),
			LZMA_PROG_ERROR);
	second.backward_size = LZMA_BACKWARD_SIZE_MIN;

	// Backward Sizes must be equal
	second.backward_size = first.backward_size + 4;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second),
			LZMA_DATA_ERROR);

	// Should fail if Backward Sizes are > LZMA_BACKWARD_SIZE_MAX
	// even though they are equal
	first.backward_size = LZMA_BACKWARD_SIZE_MAX + 1;
	second.backward_size = LZMA_BACKWARD_SIZE_MAX + 1;
	assert_lzma_ret(lzma_stream_flags_compare(&first, &second),
			LZMA_PROG_ERROR);
}


extern int
main(int argc, char **argv)
{
	tuktest_start(argc, argv);
	tuktest_run(test_lzma_stream_header_encode);
	tuktest_run(test_lzma_stream_footer_encode);
	tuktest_run(test_lzma_stream_header_decode);
	tuktest_run(test_lzma_stream_footer_decode);
	tuktest_run(test_lzma_stream_flags_compare);
	return tuktest_end();
}
