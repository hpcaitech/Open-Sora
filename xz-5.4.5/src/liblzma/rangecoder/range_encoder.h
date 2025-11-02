///////////////////////////////////////////////////////////////////////////////
//
/// \file       range_encoder.h
/// \brief      Range Encoder
///
//  Authors:    Igor Pavlov
//              Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef LZMA_RANGE_ENCODER_H
#define LZMA_RANGE_ENCODER_H

#include "range_common.h"
#include "price.h"


/// Maximum number of symbols that can be put pending into lzma_range_encoder
/// structure between calls to lzma_rc_encode(). For LZMA, 48+5 is enough
/// (match with big distance and length followed by range encoder flush).
#define RC_SYMBOLS_MAX 53


typedef struct {
	uint64_t low;
	uint64_t cache_size;
	uint32_t range;
	uint8_t cache;

	/// Number of bytes written out by rc_encode() -> rc_shift_low()
	uint64_t out_total;

	/// Number of symbols in the tables
	size_t count;

	/// rc_encode()'s position in the tables
	size_t pos;

	/// Symbols to encode
	enum {
		RC_BIT_0,
		RC_BIT_1,
		RC_DIRECT_0,
		RC_DIRECT_1,
		RC_FLUSH,
	} symbols[RC_SYMBOLS_MAX];

	/// Probabilities associated with RC_BIT_0 or RC_BIT_1
	probability *probs[RC_SYMBOLS_MAX];

} lzma_range_encoder;


static inline void
rc_reset(lzma_range_encoder *rc)
{
	rc->low = 0;
	rc->cache_size = 1;
	rc->range = UINT32_MAX;
	rc->cache = 0;
	rc->out_total = 0;
	rc->count = 0;
	rc->pos = 0;
}


static inline void
rc_forget(lzma_range_encoder *rc)
{
	// This must not be called when rc_encode() is partially done.
	assert(rc->pos == 0);
	rc->count = 0;
}


static inline void
rc_bit(lzma_range_encoder *rc, probability *prob, uint32_t bit)
{
	rc->symbols[rc->count] = bit;
	rc->probs[rc->count] = prob;
	++rc->count;
}


static inline void
rc_bittree(lzma_range_encoder *rc, probability *probs,
		uint32_t bit_count, uint32_t symbol)
{
	uint32_t model_index = 1;

	do {
		const uint32_t bit = (symbol >> --bit_count) & 1;
		rc_bit(rc, &probs[model_index], bit);
		model_index = (model_index << 1) + bit;
	} while (bit_count != 0);
}


static inline void
rc_bittree_reverse(lzma_range_encoder *rc, probability *probs,
		uint32_t bit_count, uint32_t symbol)
{
	uint32_t model_index = 1;

	do {
		const uint32_t bit = symbol & 1;
		symbol >>= 1;
		rc_bit(rc, &probs[model_index], bit);
		model_index = (model_index << 1) + bit;
	} while (--bit_count != 0);
}


static inline void
rc_direct(lzma_range_encoder *rc,
		uint32_t value, uint32_t bit_count)
{
	do {
		rc->symbols[rc->count++]
				= RC_DIRECT_0 + ((value >> --bit_count) & 1);
	} while (bit_count != 0);
}


static inline void
rc_flush(lzma_range_encoder *rc)
{
	for (size_t i = 0; i < 5; ++i)
		rc->symbols[rc->count++] = RC_FLUSH;
}


static inline bool
rc_shift_low(lzma_range_encoder *rc,
		uint8_t *out, size_t *out_pos, size_t out_size)
{
	if ((uint32_t)(rc->low) < (uint32_t)(0xFF000000)
			|| (uint32_t)(rc->low >> 32) != 0) {
		do {
			if (*out_pos == out_size)
				return true;

			out[*out_pos] = rc->cache + (uint8_t)(rc->low >> 32);
			++*out_pos;
			++rc->out_total;
			rc->cache = 0xFF;

		} while (--rc->cache_size != 0);

		rc->cache = (rc->low >> 24) & 0xFF;
	}

	++rc->cache_size;
	rc->low = (rc->low & 0x00FFFFFF) << RC_SHIFT_BITS;

	return false;
}


// NOTE: The last two arguments are uint64_t instead of size_t because in
// the dummy version these refer to the size of the whole range-encoded
// output stream, not just to the currently available output buffer space.
static inline bool
rc_shift_low_dummy(uint64_t *low, uint64_t *cache_size, uint8_t *cache,
		uint64_t *out_pos, uint64_t out_size)
{
	if ((uint32_t)(*low) < (uint32_t)(0xFF000000)
			|| (uint32_t)(*low >> 32) != 0) {
		do {
			if (*out_pos == out_size)
				return true;

			++*out_pos;
			*cache = 0xFF;

		} while (--*cache_size != 0);

		*cache = (*low >> 24) & 0xFF;
	}

	++*cache_size;
	*low = (*low & 0x00FFFFFF) << RC_SHIFT_BITS;

	return false;
}


static inline bool
rc_encode(lzma_range_encoder *rc,
		uint8_t *out, size_t *out_pos, size_t out_size)
{
	assert(rc->count <= RC_SYMBOLS_MAX);

	while (rc->pos < rc->count) {
		// Normalize
		if (rc->range < RC_TOP_VALUE) {
			if (rc_shift_low(rc, out, out_pos, out_size))
				return true;

			rc->range <<= RC_SHIFT_BITS;
		}

		// Encode a bit
		switch (rc->symbols[rc->pos]) {
		case RC_BIT_0: {
			probability prob = *rc->probs[rc->pos];
			rc->range = (rc->range >> RC_BIT_MODEL_TOTAL_BITS)
					* prob;
			prob += (RC_BIT_MODEL_TOTAL - prob) >> RC_MOVE_BITS;
			*rc->probs[rc->pos] = prob;
			break;
		}

		case RC_BIT_1: {
			probability prob = *rc->probs[rc->pos];
			const uint32_t bound = prob * (rc->range
					>> RC_BIT_MODEL_TOTAL_BITS);
			rc->low += bound;
			rc->range -= bound;
			prob -= prob >> RC_MOVE_BITS;
			*rc->probs[rc->pos] = prob;
			break;
		}

		case RC_DIRECT_0:
			rc->range >>= 1;
			break;

		case RC_DIRECT_1:
			rc->range >>= 1;
			rc->low += rc->range;
			break;

		case RC_FLUSH:
			// Prevent further normalizations.
			rc->range = UINT32_MAX;

			// Flush the last five bytes (see rc_flush()).
			do {
				if (rc_shift_low(rc, out, out_pos, out_size))
					return true;
			} while (++rc->pos < rc->count);

			// Reset the range encoder so we are ready to continue
			// encoding if we weren't finishing the stream.
			rc_reset(rc);
			return false;

		default:
			assert(0);
			break;
		}

		++rc->pos;
	}

	rc->count = 0;
	rc->pos = 0;

	return false;
}


static inline bool
rc_encode_dummy(const lzma_range_encoder *rc, uint64_t out_limit)
{
	assert(rc->count <= RC_SYMBOLS_MAX);

	uint64_t low = rc->low;
	uint64_t cache_size = rc->cache_size;
	uint32_t range = rc->range;
	uint8_t cache = rc->cache;
	uint64_t out_pos = rc->out_total;

	size_t pos = rc->pos;

	while (true) {
		// Normalize
		if (range < RC_TOP_VALUE) {
			if (rc_shift_low_dummy(&low, &cache_size, &cache,
					&out_pos, out_limit))
				return true;

			range <<= RC_SHIFT_BITS;
		}

		// This check is here because the normalization above must
		// be done before flushing the last bytes.
		if (pos == rc->count)
			break;

		// Encode a bit
		switch (rc->symbols[pos]) {
		case RC_BIT_0: {
			probability prob = *rc->probs[pos];
			range = (range >> RC_BIT_MODEL_TOTAL_BITS)
					* prob;
			break;
		}

		case RC_BIT_1: {
			probability prob = *rc->probs[pos];
			const uint32_t bound = prob * (range
					>> RC_BIT_MODEL_TOTAL_BITS);
			low += bound;
			range -= bound;
			break;
		}

		case RC_DIRECT_0:
			range >>= 1;
			break;

		case RC_DIRECT_1:
			range >>= 1;
			low += range;
			break;

		case RC_FLUSH:
		default:
			assert(0);
			break;
		}

		++pos;
	}

	// Flush the last bytes. This isn't in rc->symbols[] so we do
	// it after the above loop to take into account the size of
	// the flushing that will be done at the end of the stream.
	for (pos = 0; pos < 5; ++pos) {
		if (rc_shift_low_dummy(&low, &cache_size,
				&cache, &out_pos, out_limit))
			return true;
	}

	return false;
}


static inline uint64_t
rc_pending(const lzma_range_encoder *rc)
{
	return rc->cache_size + 5 - 1;
}

#endif
