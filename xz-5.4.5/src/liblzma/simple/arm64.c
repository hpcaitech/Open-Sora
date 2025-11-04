///////////////////////////////////////////////////////////////////////////////
//
/// \file       arm64.c
/// \brief      Filter for ARM64 binaries
///
/// This converts ARM64 relative addresses in the BL and ADRP immediates
/// to absolute values to increase redundancy of ARM64 code.
///
/// Converting B or ADR instructions was also tested but it's not useful.
/// A majority of the jumps for the B instruction are very small (+/- 0xFF).
/// These are typical for loops and if-statements. Encoding them to their
/// absolute address reduces redundancy since many of the small relative
/// jump values are repeated, but very few of the absolute addresses are.
//
//  Authors:    Lasse Collin
//              Jia Tan
//              Igor Pavlov
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "simple_private.h"


static size_t
arm64_code(void *simple lzma_attribute((__unused__)),
		uint32_t now_pos, bool is_encoder,
		uint8_t *buffer, size_t size)
{
	size_t i;

	// Clang 14.0.6 on x86-64 makes this four times bigger and 40 % slower
	// with auto-vectorization that is enabled by default with -O2.
	// Such vectorization bloat happens with -O2 when targeting ARM64 too
	// but performance hasn't been tested.
#ifdef __clang__
#	pragma clang loop vectorize(disable)
#endif
	for (i = 0; i + 4 <= size; i += 4) {
		uint32_t pc = (uint32_t)(now_pos + i);
		uint32_t instr = read32le(buffer + i);

		if ((instr >> 26) == 0x25) {
			// BL instruction:
			// The full 26-bit immediate is converted.
			// The range is +/-128 MiB.
			//
			// Using the full range is helps quite a lot with
			// big executables. Smaller range would reduce false
			// positives in non-code sections of the input though
			// so this is a compromise that slightly favors big
			// files. With the full range only six bits of the 32
			// need to match to trigger a conversion.
			const uint32_t src = instr;
			instr = 0x94000000;

			pc >>= 2;
			if (!is_encoder)
				pc = 0U - pc;

			instr |= (src + pc) & 0x03FFFFFF;
			write32le(buffer + i, instr);

		} else if ((instr & 0x9F000000) == 0x90000000) {
			// ADRP instruction:
			// Only values in the range +/-512 MiB are converted.
			//
			// Using less than the full +/-4 GiB range reduces
			// false positives on non-code sections of the input
			// while being excellent for executables up to 512 MiB.
			// The positive effect of ADRP conversion is smaller
			// than that of BL but it also doesn't hurt so much in
			// non-code sections of input because, with +/-512 MiB
			// range, nine bits of 32 need to match to trigger a
			// conversion (two 10-bit match choices = 9 bits).
			const uint32_t src = ((instr >> 29) & 3)
					| ((instr >> 3) & 0x001FFFFC);

			// With the addition only one branch is needed to
			// check the +/- range. This is usually false when
			// processing ARM64 code so branch prediction will
			// handle it well in terms of performance.
			//
			//if ((src & 0x001E0000) != 0
			// && (src & 0x001E0000) != 0x001E0000)
			if ((src + 0x00020000) & 0x001C0000)
				continue;

			instr &= 0x9000001F;

			pc >>= 12;
			if (!is_encoder)
				pc = 0U - pc;

			const uint32_t dest = src + pc;
			instr |= (dest & 3) << 29;
			instr |= (dest & 0x0003FFFC) << 3;
			instr |= (0U - (dest & 0x00020000)) & 0x00E00000;
			write32le(buffer + i, instr);
		}
	}

	return i;
}


static lzma_ret
arm64_coder_init(lzma_next_coder *next, const lzma_allocator *allocator,
		const lzma_filter_info *filters, bool is_encoder)
{
	return lzma_simple_coder_init(next, allocator, filters,
			&arm64_code, 0, 4, 4, is_encoder);
}


#ifdef HAVE_ENCODER_ARM64
extern lzma_ret
lzma_simple_arm64_encoder_init(lzma_next_coder *next,
		const lzma_allocator *allocator,
		const lzma_filter_info *filters)
{
	return arm64_coder_init(next, allocator, filters, true);
}
#endif


#ifdef HAVE_DECODER_ARM64
extern lzma_ret
lzma_simple_arm64_decoder_init(lzma_next_coder *next,
		const lzma_allocator *allocator,
		const lzma_filter_info *filters)
{
	return arm64_coder_init(next, allocator, filters, false);
}
#endif
