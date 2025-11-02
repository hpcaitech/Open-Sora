///////////////////////////////////////////////////////////////////////////////
//
/// \file       block_decoder.c
/// \brief      Decodes .xz Blocks
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "block_decoder.h"
#include "filter_decoder.h"
#include "check.h"


typedef struct {
	enum {
		SEQ_CODE,
		SEQ_PADDING,
		SEQ_CHECK,
	} sequence;

	/// The filters in the chain; initialized with lzma_raw_decoder_init().
	lzma_next_coder next;

	/// Decoding options; we also write Compressed Size and Uncompressed
	/// Size back to this structure when the decoding has been finished.
	lzma_block *block;

	/// Compressed Size calculated while decoding
	lzma_vli compressed_size;

	/// Uncompressed Size calculated while decoding
	lzma_vli uncompressed_size;

	/// Maximum allowed Compressed Size; this takes into account the
	/// size of the Block Header and Check fields when Compressed Size
	/// is unknown.
	lzma_vli compressed_limit;

	/// Maximum allowed Uncompressed Size.
	lzma_vli uncompressed_limit;

	/// Position when reading the Check field
	size_t check_pos;

	/// Check of the uncompressed data
	lzma_check_state check;

	/// True if the integrity check won't be calculated and verified.
	bool ignore_check;
} lzma_block_coder;


static inline bool
is_size_valid(lzma_vli size, lzma_vli reference)
{
	return reference == LZMA_VLI_UNKNOWN || reference == size;
}


static lzma_ret
block_decode(void *coder_ptr, const lzma_allocator *allocator,
		const uint8_t *restrict in, size_t *restrict in_pos,
		size_t in_size, uint8_t *restrict out,
		size_t *restrict out_pos, size_t out_size, lzma_action action)
{
	lzma_block_coder *coder = coder_ptr;

	switch (coder->sequence) {
	case SEQ_CODE: {
		const size_t in_start = *in_pos;
		const size_t out_start = *out_pos;

		// Limit the amount of input and output space that we give
		// to the raw decoder based on the information we have
		// (or don't have) from Block Header.
		const size_t in_stop = *in_pos + (size_t)my_min(
			in_size - *in_pos,
			coder->compressed_limit - coder->compressed_size);
		const size_t out_stop = *out_pos + (size_t)my_min(
			out_size - *out_pos,
			coder->uncompressed_limit - coder->uncompressed_size);

		const lzma_ret ret = coder->next.code(coder->next.coder,
				allocator, in, in_pos, in_stop,
				out, out_pos, out_stop, action);

		const size_t in_used = *in_pos - in_start;
		const size_t out_used = *out_pos - out_start;

		// Because we have limited the input and output sizes,
		// we know that these cannot grow too big or overflow.
		coder->compressed_size += in_used;
		coder->uncompressed_size += out_used;

		if (ret == LZMA_OK) {
			const bool comp_done = coder->compressed_size
					== coder->block->compressed_size;
			const bool uncomp_done = coder->uncompressed_size
					== coder->block->uncompressed_size;

			// If both input and output amounts match the sizes
			// in Block Header but we still got LZMA_OK instead
			// of LZMA_STREAM_END, the file is broken.
			if (comp_done && uncomp_done)
				return LZMA_DATA_ERROR;

			// If the decoder has consumed all the input that it
			// needs but it still couldn't fill the output buffer
			// or return LZMA_STREAM_END, the file is broken.
			if (comp_done && *out_pos < out_size)
				return LZMA_DATA_ERROR;

			// If the decoder has produced all the output but
			// it still didn't return LZMA_STREAM_END or consume
			// more input (for example, detecting an end of
			// payload marker may need more input but produce
			// no output) the file is broken.
			if (uncomp_done && *in_pos < in_size)
				return LZMA_DATA_ERROR;
		}

		// Don't waste time updating the integrity check if it will be
		// ignored. Also skip it if no new output was produced. This
		// avoids null pointer + 0 (undefined behavior) when out == 0.
		if (!coder->ignore_check && out_used > 0)
			lzma_check_update(&coder->check, coder->block->check,
					out + out_start, out_used);

		if (ret != LZMA_STREAM_END)
			return ret;

		// Compressed and Uncompressed Sizes are now at their final
		// values. Verify that they match the values given to us.
		if (!is_size_valid(coder->compressed_size,
					coder->block->compressed_size)
				|| !is_size_valid(coder->uncompressed_size,
					coder->block->uncompressed_size))
			return LZMA_DATA_ERROR;

		// Copy the values into coder->block. The caller
		// may use this information to construct Index.
		coder->block->compressed_size = coder->compressed_size;
		coder->block->uncompressed_size = coder->uncompressed_size;

		coder->sequence = SEQ_PADDING;
	}

	// Fall through

	case SEQ_PADDING:
		// Compressed Data is padded to a multiple of four bytes.
		while (coder->compressed_size & 3) {
			if (*in_pos >= in_size)
				return LZMA_OK;

			// We use compressed_size here just get the Padding
			// right. The actual Compressed Size was stored to
			// coder->block already, and won't be modified by
			// us anymore.
			++coder->compressed_size;

			if (in[(*in_pos)++] != 0x00)
				return LZMA_DATA_ERROR;
		}

		if (coder->block->check == LZMA_CHECK_NONE)
			return LZMA_STREAM_END;

		if (!coder->ignore_check)
			lzma_check_finish(&coder->check, coder->block->check);

		coder->sequence = SEQ_CHECK;

	// Fall through

	case SEQ_CHECK: {
		const size_t check_size = lzma_check_size(coder->block->check);
		lzma_bufcpy(in, in_pos, in_size, coder->block->raw_check,
				&coder->check_pos, check_size);
		if (coder->check_pos < check_size)
			return LZMA_OK;

		// Validate the Check only if we support it.
		// coder->check.buffer may be uninitialized
		// when the Check ID is not supported.
		if (!coder->ignore_check
				&& lzma_check_is_supported(coder->block->check)
				&& memcmp(coder->block->raw_check,
					coder->check.buffer.u8,
					check_size) != 0)
			return LZMA_DATA_ERROR;

		return LZMA_STREAM_END;
	}
	}

	return LZMA_PROG_ERROR;
}


static void
block_decoder_end(void *coder_ptr, const lzma_allocator *allocator)
{
	lzma_block_coder *coder = coder_ptr;
	lzma_next_end(&coder->next, allocator);
	lzma_free(coder, allocator);
	return;
}


extern lzma_ret
lzma_block_decoder_init(lzma_next_coder *next, const lzma_allocator *allocator,
		lzma_block *block)
{
	lzma_next_coder_init(&lzma_block_decoder_init, next, allocator);

	// Validate the options. lzma_block_unpadded_size() does that for us
	// except for Uncompressed Size and filters. Filters are validated
	// by the raw decoder.
	if (lzma_block_unpadded_size(block) == 0
			|| !lzma_vli_is_valid(block->uncompressed_size))
		return LZMA_PROG_ERROR;

	// Allocate *next->coder if needed.
	lzma_block_coder *coder = next->coder;
	if (coder == NULL) {
		coder = lzma_alloc(sizeof(lzma_block_coder), allocator);
		if (coder == NULL)
			return LZMA_MEM_ERROR;

		next->coder = coder;
		next->code = &block_decode;
		next->end = &block_decoder_end;
		coder->next = LZMA_NEXT_CODER_INIT;
	}

	// Basic initializations
	coder->sequence = SEQ_CODE;
	coder->block = block;
	coder->compressed_size = 0;
	coder->uncompressed_size = 0;

	// If Compressed Size is not known, we calculate the maximum allowed
	// value so that encoded size of the Block (including Block Padding)
	// is still a valid VLI and a multiple of four.
	coder->compressed_limit
			= block->compressed_size == LZMA_VLI_UNKNOWN
				? (LZMA_VLI_MAX & ~LZMA_VLI_C(3))
					- block->header_size
					- lzma_check_size(block->check)
				: block->compressed_size;

	// With Uncompressed Size this is simpler. If Block Header lacks
	// the size info, then LZMA_VLI_MAX is the maximum possible
	// Uncompressed Size.
	coder->uncompressed_limit
			= block->uncompressed_size == LZMA_VLI_UNKNOWN
				? LZMA_VLI_MAX
				: block->uncompressed_size;

	// Initialize the check. It's caller's problem if the Check ID is not
	// supported, and the Block decoder cannot verify the Check field.
	// Caller can test lzma_check_is_supported(block->check).
	coder->check_pos = 0;
	lzma_check_init(&coder->check, block->check);

	coder->ignore_check = block->version >= 1
			? block->ignore_check : false;

	// Initialize the filter chain.
	return lzma_raw_decoder_init(&coder->next, allocator,
			block->filters);
}


extern LZMA_API(lzma_ret)
lzma_block_decoder(lzma_stream *strm, lzma_block *block)
{
	lzma_next_strm_init(lzma_block_decoder_init, strm, block);

	strm->internal->supported_actions[LZMA_RUN] = true;
	strm->internal->supported_actions[LZMA_FINISH] = true;

	return LZMA_OK;
}
