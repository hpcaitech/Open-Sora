///////////////////////////////////////////////////////////////////////////////
//
/// \file       lzip_decoder.c
/// \brief      Decodes .lz (lzip) files
//
//  Author:     Michał Górny
//              Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "lzip_decoder.h"
#include "lzma_decoder.h"
#include "check.h"


// .lz format version 0 lacks the 64-bit Member size field in the footer.
#define LZIP_V0_FOOTER_SIZE 12
#define LZIP_V1_FOOTER_SIZE 20
#define LZIP_FOOTER_SIZE_MAX LZIP_V1_FOOTER_SIZE

// lc/lp/pb are hardcoded in the .lz format.
#define LZIP_LC 3
#define LZIP_LP 0
#define LZIP_PB 2


typedef struct {
	enum {
		SEQ_ID_STRING,
		SEQ_VERSION,
		SEQ_DICT_SIZE,
		SEQ_CODER_INIT,
		SEQ_LZMA_STREAM,
		SEQ_MEMBER_FOOTER,
	} sequence;

	/// .lz member format version
	uint32_t version;

	/// CRC32 of the uncompressed data in the .lz member
	uint32_t crc32;

	/// Uncompressed size of the .lz member
	uint64_t uncompressed_size;

	/// Compressed size of the .lz member
	uint64_t member_size;

	/// Memory usage limit
	uint64_t memlimit;

	/// Amount of memory actually needed
	uint64_t memusage;

	/// If true, LZMA_GET_CHECK is returned after decoding the header
	/// fields. As all files use CRC32 this is redundant but it's
	/// implemented anyway since the initialization functions supports
	/// all other flags in addition to LZMA_TELL_ANY_CHECK.
	bool tell_any_check;

	/// If true, we won't calculate or verify the CRC32 of
	/// the uncompressed data.
	bool ignore_check;

	/// If true, we will decode concatenated .lz members and stop if
	/// non-.lz data is seen after at least one member has been
	/// successfully decoded.
	bool concatenated;

	/// When decoding concatenated .lz members, this is true as long as
	/// we are decoding the first .lz member. This is needed to avoid
	/// incorrect LZMA_FORMAT_ERROR in case there is non-.lz data at
	/// the end of the file.
	bool first_member;

	/// Reading position in the header and footer fields
	size_t pos;

	/// Buffer to hold the .lz footer fields
	uint8_t buffer[LZIP_FOOTER_SIZE_MAX];

	/// Options decoded from the .lz header that needed to initialize
	/// the LZMA1 decoder.
	lzma_options_lzma options;

	/// LZMA1 decoder
	lzma_next_coder lzma_decoder;

} lzma_lzip_coder;


static lzma_ret
lzip_decode(void *coder_ptr, const lzma_allocator *allocator,
		const uint8_t *restrict in, size_t *restrict in_pos,
		size_t in_size, uint8_t *restrict out,
		size_t *restrict out_pos, size_t out_size, lzma_action action)
{
	lzma_lzip_coder *coder = coder_ptr;

	while (true)
	switch (coder->sequence) {
	case SEQ_ID_STRING: {
		// The "ID string" or magic bytes are "LZIP" in US-ASCII.
		const uint8_t lzip_id_string[4] = { 0x4C, 0x5A, 0x49, 0x50 };

		while (coder->pos < sizeof(lzip_id_string)) {
			if (*in_pos >= in_size) {
				// If we are on the 2nd+ concatenated member
				// and the input ends before we can read
				// the magic bytes, we discard the bytes that
				// were already read (up to 3) and finish.
				// See the reasoning below.
				return !coder->first_member
						&& action == LZMA_FINISH
					? LZMA_STREAM_END : LZMA_OK;
			}

			if (in[*in_pos] != lzip_id_string[coder->pos]) {
				// The .lz format allows putting non-.lz data
				// at the end of the file. If we have seen
				// at least one valid .lz member already,
				// then we won't consume the byte at *in_pos
				// and will return LZMA_STREAM_END. This way
				// apps can easily locate and read the non-.lz
				// data after the .lz member(s).
				//
				// NOTE: If the first 1-3 bytes of the non-.lz
				// data match the .lz ID string then the first
				// 1-3 bytes of the junk will get ignored by
				// us. If apps want to properly locate the
				// trailing data they must ensure that the
				// first byte of their custom data isn't the
				// same as the first byte of .lz ID string.
				// With the liblzma API we cannot rewind the
				// input position across calls to lzma_code().
				return !coder->first_member
					? LZMA_STREAM_END : LZMA_FORMAT_ERROR;
			}

			++*in_pos;
			++coder->pos;
		}

		coder->pos = 0;

		coder->crc32 = 0;
		coder->uncompressed_size = 0;
		coder->member_size = sizeof(lzip_id_string);

		coder->sequence = SEQ_VERSION;
	}

	// Fall through

	case SEQ_VERSION:
		if (*in_pos >= in_size)
			return LZMA_OK;

		coder->version = in[(*in_pos)++];

		// We support version 0 and unextended version 1.
		if (coder->version > 1)
			return LZMA_OPTIONS_ERROR;

		++coder->member_size;
		coder->sequence = SEQ_DICT_SIZE;

		// .lz versions 0 and 1 use CRC32 as the integrity check
		// so if the application wanted to know that
		// (LZMA_TELL_ANY_CHECK) we can tell it now.
		if (coder->tell_any_check)
			return LZMA_GET_CHECK;

	// Fall through

	case SEQ_DICT_SIZE: {
		if (*in_pos >= in_size)
			return LZMA_OK;

		const uint32_t ds = in[(*in_pos)++];
		++coder->member_size;

		// The five lowest bits are for the base-2 logarithm of
		// the dictionary size and the highest three bits are
		// the fractional part (0/16 to 7/16) that will be
		// subtracted to get the final value.
		//
		// For example, with 0xB5:
		//     b2log = 21
		//     fracnum = 5
		//     dict_size = 2^21 - 2^21 * 5 / 16 = 1408 KiB
		const uint32_t b2log = ds & 0x1F;
		const uint32_t fracnum = ds >> 5;

		// The format versions 0 and 1 allow dictionary size in the
		// range [4 KiB, 512 MiB].
		if (b2log < 12 || b2log > 29 || (b2log == 12 && fracnum > 0))
			return LZMA_DATA_ERROR;

		//   2^[b2log] - 2^[b2log] * [fracnum] / 16
		// = 2^[b2log] - [fracnum] * 2^([b2log] - 4)
		coder->options.dict_size = (UINT32_C(1) << b2log)
				- (fracnum << (b2log - 4));

		assert(coder->options.dict_size >= 4096);
		assert(coder->options.dict_size <= (UINT32_C(512) << 20));

		coder->options.preset_dict = NULL;
		coder->options.lc = LZIP_LC;
		coder->options.lp = LZIP_LP;
		coder->options.pb = LZIP_PB;

		// Calculate the memory usage.
		coder->memusage = lzma_lzma_decoder_memusage(&coder->options)
				+ LZMA_MEMUSAGE_BASE;

		// Initialization is a separate step because if we return
		// LZMA_MEMLIMIT_ERROR we need to be able to restart after
		// the memlimit has been increased.
		coder->sequence = SEQ_CODER_INIT;
	}

	// Fall through

	case SEQ_CODER_INIT: {
		if (coder->memusage > coder->memlimit)
			return LZMA_MEMLIMIT_ERROR;

		const lzma_filter_info filters[2] = {
			{
				.id = LZMA_FILTER_LZMA1,
				.init = &lzma_lzma_decoder_init,
				.options = &coder->options,
			}, {
				.init = NULL,
			}
		};

		return_if_error(lzma_next_filter_init(&coder->lzma_decoder,
				allocator, filters));

		coder->crc32 = 0;
		coder->sequence = SEQ_LZMA_STREAM;
	}

	// Fall through

	case SEQ_LZMA_STREAM: {
		const size_t in_start = *in_pos;
		const size_t out_start = *out_pos;

		const lzma_ret ret = coder->lzma_decoder.code(
				coder->lzma_decoder.coder, allocator,
				in, in_pos, in_size, out, out_pos, out_size,
				action);

		const size_t out_used = *out_pos - out_start;

		coder->member_size += *in_pos - in_start;
		coder->uncompressed_size += out_used;

		// Don't update the CRC32 if the integrity check will be
		// ignored or if there was no new output. The latter is
		// important in case out == NULL to avoid null pointer + 0
		// which is undefined behavior.
		if (!coder->ignore_check && out_used > 0)
			coder->crc32 = lzma_crc32(out + out_start, out_used,
					coder->crc32);

		if (ret != LZMA_STREAM_END)
			return ret;

		coder->sequence = SEQ_MEMBER_FOOTER;
	}

	// Fall through

	case SEQ_MEMBER_FOOTER: {
		// The footer of .lz version 0 lacks the Member size field.
		// This is the only difference between version 0 and
		// unextended version 1 formats.
		const size_t footer_size = coder->version == 0
				? LZIP_V0_FOOTER_SIZE
				: LZIP_V1_FOOTER_SIZE;

		// Copy the CRC32, Data size, and Member size fields to
		// the internal buffer.
		lzma_bufcpy(in, in_pos, in_size, coder->buffer, &coder->pos,
				footer_size);

		// Return if we didn't get the whole footer yet.
		if (coder->pos < footer_size)
			return LZMA_OK;

		coder->pos = 0;
		coder->member_size += footer_size;

		// Check that the footer fields match the observed data.
		if (!coder->ignore_check
				&& coder->crc32 != read32le(&coder->buffer[0]))
			return LZMA_DATA_ERROR;

		if (coder->uncompressed_size != read64le(&coder->buffer[4]))
			return LZMA_DATA_ERROR;

		if (coder->version > 0) {
			// .lz version 0 has no Member size field.
			if (coder->member_size != read64le(&coder->buffer[12]))
				return LZMA_DATA_ERROR;
		}

		// Decoding is finished if we weren't requested to decode
		// more than one .lz member.
		if (!coder->concatenated)
			return LZMA_STREAM_END;

		coder->first_member = false;
		coder->sequence = SEQ_ID_STRING;
		break;
	}

	default:
		assert(0);
		return LZMA_PROG_ERROR;
	}

	// Never reached
}


static void
lzip_decoder_end(void *coder_ptr, const lzma_allocator *allocator)
{
	lzma_lzip_coder *coder = coder_ptr;
	lzma_next_end(&coder->lzma_decoder, allocator);
	lzma_free(coder, allocator);
	return;
}


static lzma_check
lzip_decoder_get_check(const void *coder_ptr lzma_attribute((__unused__)))
{
	return LZMA_CHECK_CRC32;
}


static lzma_ret
lzip_decoder_memconfig(void *coder_ptr, uint64_t *memusage,
		uint64_t *old_memlimit, uint64_t new_memlimit)
{
	lzma_lzip_coder *coder = coder_ptr;

	*memusage = coder->memusage;
	*old_memlimit = coder->memlimit;

	if (new_memlimit != 0) {
		if (new_memlimit < coder->memusage)
			return LZMA_MEMLIMIT_ERROR;

		coder->memlimit = new_memlimit;
	}

	return LZMA_OK;
}


extern lzma_ret
lzma_lzip_decoder_init(
		lzma_next_coder *next, const lzma_allocator *allocator,
		uint64_t memlimit, uint32_t flags)
{
	lzma_next_coder_init(&lzma_lzip_decoder_init, next, allocator);

	if (flags & ~LZMA_SUPPORTED_FLAGS)
		return LZMA_OPTIONS_ERROR;

	lzma_lzip_coder *coder = next->coder;
	if (coder == NULL) {
		coder = lzma_alloc(sizeof(lzma_lzip_coder), allocator);
		if (coder == NULL)
			return LZMA_MEM_ERROR;

		next->coder = coder;
		next->code = &lzip_decode;
		next->end = &lzip_decoder_end;
		next->get_check = &lzip_decoder_get_check;
		next->memconfig = &lzip_decoder_memconfig;

		coder->lzma_decoder = LZMA_NEXT_CODER_INIT;
	}

	coder->sequence = SEQ_ID_STRING;
	coder->memlimit = my_max(1, memlimit);
	coder->memusage = LZMA_MEMUSAGE_BASE;
	coder->tell_any_check = (flags & LZMA_TELL_ANY_CHECK) != 0;
	coder->ignore_check = (flags & LZMA_IGNORE_CHECK) != 0;
	coder->concatenated = (flags & LZMA_CONCATENATED) != 0;
	coder->first_member = true;
	coder->pos = 0;

	return LZMA_OK;
}


extern LZMA_API(lzma_ret)
lzma_lzip_decoder(lzma_stream *strm, uint64_t memlimit, uint32_t flags)
{
	lzma_next_strm_init(lzma_lzip_decoder_init, strm, memlimit, flags);

	strm->internal->supported_actions[LZMA_RUN] = true;
	strm->internal->supported_actions[LZMA_FINISH] = true;

	return LZMA_OK;
}
