///////////////////////////////////////////////////////////////////////////////
//
/// \file       microlzma_decoder.c
/// \brief      Decode MicroLZMA format
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "lzma_decoder.h"
#include "lz_decoder.h"


typedef struct {
	/// LZMA1 decoder
	lzma_next_coder lzma;

	/// Compressed size of the stream as given by the application.
	/// This must be exactly correct.
	///
	/// This will be decremented when input is read.
	uint64_t comp_size;

	/// Uncompressed size of the stream as given by the application.
	/// This may be less than the actual uncompressed size if
	/// uncomp_size_is_exact is false.
	///
	/// This will be decremented when output is produced.
	lzma_vli uncomp_size;

	/// LZMA dictionary size as given by the application
	uint32_t dict_size;

	/// If true, the exact uncompressed size is known. If false,
	/// uncomp_size may be smaller than the real uncompressed size;
	/// uncomp_size may never be bigger than the real uncompressed size.
	bool uncomp_size_is_exact;

	/// True once the first byte of the MicroLZMA stream
	/// has been processed.
	bool props_decoded;
} lzma_microlzma_coder;


static lzma_ret
microlzma_decode(void *coder_ptr, const lzma_allocator *allocator,
		const uint8_t *restrict in, size_t *restrict in_pos,
		size_t in_size, uint8_t *restrict out,
		size_t *restrict out_pos, size_t out_size, lzma_action action)
{
	lzma_microlzma_coder *coder = coder_ptr;

	// Remember the in start position so that we can update comp_size.
	const size_t in_start = *in_pos;

	// Remember the out start position so that we can update uncomp_size.
	const size_t out_start = *out_pos;

	// Limit the amount of input so that the decoder won't read more than
	// comp_size. This is required when uncomp_size isn't exact because
	// in that case the LZMA decoder will try to decode more input even
	// when it has no output space (it can be looking for EOPM).
	if (in_size - *in_pos > coder->comp_size)
		in_size = *in_pos + (size_t)(coder->comp_size);

	// When the exact uncompressed size isn't known, we must limit
	// the available output space to prevent the LZMA decoder from
	// trying to decode too much.
	if (!coder->uncomp_size_is_exact
			&& out_size - *out_pos > coder->uncomp_size)
		out_size = *out_pos + (size_t)(coder->uncomp_size);

	if (!coder->props_decoded) {
		// There must be at least one byte of input to decode
		// the properties byte.
		if (*in_pos >= in_size)
			return LZMA_OK;

		lzma_options_lzma options = {
			.dict_size = coder->dict_size,
			.preset_dict = NULL,
			.preset_dict_size = 0,
			.ext_flags = 0, // EOPM not allowed when size is known
			.ext_size_low = UINT32_MAX, // Unknown size by default
			.ext_size_high = UINT32_MAX,
		};

		if (coder->uncomp_size_is_exact)
			lzma_set_ext_size(options, coder->uncomp_size);

		// The properties are stored as bitwise-negation
		// of the typical encoding.
		if (lzma_lzma_lclppb_decode(&options, ~in[*in_pos]))
			return LZMA_OPTIONS_ERROR;

		++*in_pos;

		// Initialize the decoder.
		lzma_filter_info filters[2] = {
			{
				.id = LZMA_FILTER_LZMA1EXT,
				.init = &lzma_lzma_decoder_init,
				.options = &options,
			}, {
				.init = NULL,
			}
		};

		return_if_error(lzma_next_filter_init(&coder->lzma,
				allocator, filters));

		// Pass one dummy 0x00 byte to the LZMA decoder since that
		// is what it expects the first byte to be.
		const uint8_t dummy_in = 0;
		size_t dummy_in_pos = 0;
		if (coder->lzma.code(coder->lzma.coder, allocator,
				&dummy_in, &dummy_in_pos, 1,
				out, out_pos, out_size, LZMA_RUN) != LZMA_OK)
			return LZMA_PROG_ERROR;

		assert(dummy_in_pos == 1);
		coder->props_decoded = true;
	}

	// The rest is normal LZMA decoding.
	lzma_ret ret = coder->lzma.code(coder->lzma.coder, allocator,
				in, in_pos, in_size,
				out, out_pos, out_size, action);

	// Update the remaining compressed size.
	assert(coder->comp_size >= *in_pos - in_start);
	coder->comp_size -= *in_pos - in_start;

	if (coder->uncomp_size_is_exact) {
		// After successful decompression of the complete stream
		// the compressed size must match.
		if (ret == LZMA_STREAM_END && coder->comp_size != 0)
			ret = LZMA_DATA_ERROR;
	} else {
		// Update the amount of output remaining.
		assert(coder->uncomp_size >= *out_pos - out_start);
		coder->uncomp_size -= *out_pos - out_start;

		// - We must not get LZMA_STREAM_END because the stream
		//   shouldn't have EOPM.
		// - We must use uncomp_size to determine when to
		//   return LZMA_STREAM_END.
		if (ret == LZMA_STREAM_END)
			ret = LZMA_DATA_ERROR;
		else if (coder->uncomp_size == 0)
			ret = LZMA_STREAM_END;
	}

	return ret;
}


static void
microlzma_decoder_end(void *coder_ptr, const lzma_allocator *allocator)
{
	lzma_microlzma_coder *coder = coder_ptr;
	lzma_next_end(&coder->lzma, allocator);
	lzma_free(coder, allocator);
	return;
}


static lzma_ret
microlzma_decoder_init(lzma_next_coder *next, const lzma_allocator *allocator,
		uint64_t comp_size,
		uint64_t uncomp_size, bool uncomp_size_is_exact,
		uint32_t dict_size)
{
	lzma_next_coder_init(&microlzma_decoder_init, next, allocator);

	lzma_microlzma_coder *coder = next->coder;

	if (coder == NULL) {
		coder = lzma_alloc(sizeof(lzma_microlzma_coder), allocator);
		if (coder == NULL)
			return LZMA_MEM_ERROR;

		next->coder = coder;
		next->code = &microlzma_decode;
		next->end = &microlzma_decoder_end;

		coder->lzma = LZMA_NEXT_CODER_INIT;
	}

	// The public API is uint64_t but the internal LZ decoder API uses
	// lzma_vli.
	if (uncomp_size > LZMA_VLI_MAX)
		return LZMA_OPTIONS_ERROR;

	coder->comp_size = comp_size;
	coder->uncomp_size = uncomp_size;
	coder->uncomp_size_is_exact = uncomp_size_is_exact;
	coder->dict_size = dict_size;

	coder->props_decoded = false;

	return LZMA_OK;
}


extern LZMA_API(lzma_ret)
lzma_microlzma_decoder(lzma_stream *strm, uint64_t comp_size,
		uint64_t uncomp_size, lzma_bool uncomp_size_is_exact,
		uint32_t dict_size)
{
	lzma_next_strm_init(microlzma_decoder_init, strm, comp_size,
			uncomp_size, uncomp_size_is_exact, dict_size);

	strm->internal->supported_actions[LZMA_RUN] = true;
	strm->internal->supported_actions[LZMA_FINISH] = true;

	return LZMA_OK;
}
