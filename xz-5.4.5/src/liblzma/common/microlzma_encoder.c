///////////////////////////////////////////////////////////////////////////////
//
/// \file       microlzma_encoder.c
/// \brief      Encode into MicroLZMA format
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "lzma_encoder.h"


typedef struct {
	/// LZMA1 encoder
	lzma_next_coder lzma;

	/// LZMA properties byte (lc/lp/pb)
	uint8_t props;
} lzma_microlzma_coder;


static lzma_ret
microlzma_encode(void *coder_ptr, const lzma_allocator *allocator,
		const uint8_t *restrict in, size_t *restrict in_pos,
		size_t in_size, uint8_t *restrict out,
		size_t *restrict out_pos, size_t out_size, lzma_action action)
{
	lzma_microlzma_coder *coder = coder_ptr;

	// Remember *out_pos so that we can overwrite the first byte with
	// the LZMA properties byte.
	const size_t out_start = *out_pos;

	// Remember *in_pos so that we can set it based on how many
	// uncompressed bytes were actually encoded.
	const size_t in_start = *in_pos;

	// Set the output size limit based on the available output space.
	// We know that the encoder supports set_out_limit() so
	// LZMA_OPTIONS_ERROR isn't possible. LZMA_BUF_ERROR is possible
	// but lzma_code() has an assertion to not allow it to be returned
	// from here and I don't want to change that for now, so
	// LZMA_BUF_ERROR becomes LZMA_PROG_ERROR.
	uint64_t uncomp_size;
	if (coder->lzma.set_out_limit(coder->lzma.coder,
			&uncomp_size, out_size - *out_pos) != LZMA_OK)
		return LZMA_PROG_ERROR;

	// set_out_limit fails if this isn't true.
	assert(out_size - *out_pos >= 6);

	// Encode as much as possible.
	const lzma_ret ret = coder->lzma.code(coder->lzma.coder, allocator,
			in, in_pos, in_size, out, out_pos, out_size, action);

	if (ret != LZMA_STREAM_END) {
		if (ret == LZMA_OK) {
			assert(0);
			return LZMA_PROG_ERROR;
		}

		return ret;
	}

	// The first output byte is bitwise-negation of the properties byte.
	// We know that there is space for this byte because set_out_limit
	// and the actual encoding succeeded.
	out[out_start] = (uint8_t)(~coder->props);

	// The LZMA encoder likely read more input than it was able to encode.
	// Set *in_pos based on uncomp_size.
	assert(uncomp_size <= in_size - in_start);
	*in_pos = in_start + (size_t)(uncomp_size);

	return ret;
}


static void
microlzma_encoder_end(void *coder_ptr, const lzma_allocator *allocator)
{
	lzma_microlzma_coder *coder = coder_ptr;
	lzma_next_end(&coder->lzma, allocator);
	lzma_free(coder, allocator);
	return;
}


static lzma_ret
microlzma_encoder_init(lzma_next_coder *next, const lzma_allocator *allocator,
		const lzma_options_lzma *options)
{
	lzma_next_coder_init(&microlzma_encoder_init, next, allocator);

	lzma_microlzma_coder *coder = next->coder;

	if (coder == NULL) {
		coder = lzma_alloc(sizeof(lzma_microlzma_coder), allocator);
		if (coder == NULL)
			return LZMA_MEM_ERROR;

		next->coder = coder;
		next->code = &microlzma_encode;
		next->end = &microlzma_encoder_end;

		coder->lzma = LZMA_NEXT_CODER_INIT;
	}

	// Encode the properties byte. Bitwise-negation of it will be the
	// first output byte.
	if (lzma_lzma_lclppb_encode(options, &coder->props))
		return LZMA_OPTIONS_ERROR;

	// Initialize the LZMA encoder.
	const lzma_filter_info filters[2] = {
		{
			.id = LZMA_FILTER_LZMA1,
			.init = &lzma_lzma_encoder_init,
			.options = (void *)(options),
		}, {
			.init = NULL,
		}
	};

	return lzma_next_filter_init(&coder->lzma, allocator, filters);
}


extern LZMA_API(lzma_ret)
lzma_microlzma_encoder(lzma_stream *strm, const lzma_options_lzma *options)
{
	lzma_next_strm_init(microlzma_encoder_init, strm, options);

	strm->internal->supported_actions[LZMA_FINISH] = true;

	return LZMA_OK;

}
