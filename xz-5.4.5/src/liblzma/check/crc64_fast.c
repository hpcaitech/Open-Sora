///////////////////////////////////////////////////////////////////////////////
//
/// \file       crc64.c
/// \brief      CRC64 calculation
///
/// There are two methods in this file. crc64_generic uses the
/// the slice-by-four algorithm. This is the same idea that is
/// used in crc32_fast.c, but for CRC64 we use only four tables
/// instead of eight to avoid increasing CPU cache usage.
///
/// crc64_clmul uses 32/64-bit x86 SSSE3, SSE4.1, and CLMUL instructions.
/// It was derived from
/// https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/fast-crc-computation-generic-polynomials-pclmulqdq-paper.pdf
/// and the public domain code from https://github.com/rawrunprotected/crc
/// (URLs were checked on 2022-11-07).
///
/// FIXME: Builds for 32-bit x86 use crc64_x86.S by default instead
/// of this file and thus CLMUL version isn't available on 32-bit x86
/// unless configured with --disable-assembler. Even then the lookup table
/// isn't omitted in crc64_table.c since it doesn't know that assembly
/// code has been disabled.
//
//  Authors:    Lasse Collin
//              Ilya Kurdyukov
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "check.h"

#undef CRC_GENERIC
#undef CRC_CLMUL
#undef CRC_USE_GENERIC_FOR_SMALL_INPUTS

// If CLMUL cannot be used then only the generic slice-by-four is built.
#if !defined(HAVE_USABLE_CLMUL)
#	define CRC_GENERIC 1

// If CLMUL is allowed unconditionally in the compiler options then the
// generic version can be omitted. Note that this doesn't work with MSVC
// as I don't know how to detect the features here.
//
// NOTE: Keep this this in sync with crc64_table.c.
#elif (defined(__SSSE3__) && defined(__SSE4_1__) && defined(__PCLMUL__)) \
		|| (defined(__e2k__) && __iset__ >= 6)
#	define CRC_CLMUL 1

// Otherwise build both and detect at runtime which version to use.
#else
#	define CRC_GENERIC 1
#	define CRC_CLMUL 1

/*
	// The generic code is much faster with 1-8-byte inputs and has
	// similar performance up to 16 bytes  at least in microbenchmarks
	// (it depends on input buffer alignment too). If both versions are
	// built, this #define will use the generic version for inputs up to
	// 16 bytes and CLMUL for bigger inputs. It saves a little in code
	// size since the special cases for 0-16-byte inputs will be omitted
	// from the CLMUL code.
#	define CRC_USE_GENERIC_FOR_SMALL_INPUTS 1
*/

#	if defined(_MSC_VER)
#		include <intrin.h>
#	elif defined(HAVE_CPUID_H)
#		include <cpuid.h>
#	endif
#endif


/////////////////////////////////
// Generic slice-by-four CRC64 //
/////////////////////////////////

#ifdef CRC_GENERIC

#include "crc_macros.h"


#ifdef WORDS_BIGENDIAN
#	define A1(x) ((x) >> 56)
#else
#	define A1 A
#endif


// See the comments in crc32_fast.c. They aren't duplicated here.
static uint64_t
crc64_generic(const uint8_t *buf, size_t size, uint64_t crc)
{
	crc = ~crc;

#ifdef WORDS_BIGENDIAN
	crc = bswap64(crc);
#endif

	if (size > 4) {
		while ((uintptr_t)(buf) & 3) {
			crc = lzma_crc64_table[0][*buf++ ^ A1(crc)] ^ S8(crc);
			--size;
		}

		const uint8_t *const limit = buf + (size & ~(size_t)(3));
		size &= (size_t)(3);

		while (buf < limit) {
#ifdef WORDS_BIGENDIAN
			const uint32_t tmp = (uint32_t)(crc >> 32)
					^ aligned_read32ne(buf);
#else
			const uint32_t tmp = (uint32_t)crc
					^ aligned_read32ne(buf);
#endif
			buf += 4;

			crc = lzma_crc64_table[3][A(tmp)]
			    ^ lzma_crc64_table[2][B(tmp)]
			    ^ S32(crc)
			    ^ lzma_crc64_table[1][C(tmp)]
			    ^ lzma_crc64_table[0][D(tmp)];
		}
	}

	while (size-- != 0)
		crc = lzma_crc64_table[0][*buf++ ^ A1(crc)] ^ S8(crc);

#ifdef WORDS_BIGENDIAN
	crc = bswap64(crc);
#endif

	return ~crc;
}
#endif


/////////////////////
// x86 CLMUL CRC64 //
/////////////////////

#ifdef CRC_CLMUL

#include <immintrin.h>


/*
// These functions were used to generate the constants
// at the top of crc64_clmul().
static uint64_t
calc_lo(uint64_t poly)
{
	uint64_t a = poly;
	uint64_t b = 0;

	for (unsigned i = 0; i < 64; ++i) {
		b = (b >> 1) | (a << 63);
		a = (a >> 1) ^ (a & 1 ? poly : 0);
	}

	return b;
}

static uint64_t
calc_hi(uint64_t poly, uint64_t a)
{
	for (unsigned i = 0; i < 64; ++i)
		a = (a >> 1) ^ (a & 1 ? poly : 0);

	return a;
}
*/


#define MASK_L(in, mask, r) \
	r = _mm_shuffle_epi8(in, mask)

#define MASK_H(in, mask, r) \
	r = _mm_shuffle_epi8(in, _mm_xor_si128(mask, vsign))

#define MASK_LH(in, mask, low, high) \
	MASK_L(in, mask, low); \
	MASK_H(in, mask, high)


// MSVC (VS2015 - VS2022) produces bad 32-bit x86 code from the CLMUL CRC
// code when optimizations are enabled (release build). According to the bug
// report, the ebx register is corrupted and the calculated result is wrong.
// Trying to workaround the problem with "__asm mov ebx, ebx" didn't help.
// The following pragma works and performance is still good. x86-64 builds
// aren't affected by this problem.
//
// NOTE: Another pragma after the function restores the optimizations.
// If the #if condition here is updated, the other one must be updated too.
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER) && !defined(__clang__) \
		&& defined(_M_IX86)
#	pragma optimize("g", off)
#endif

// EDG-based compilers (Intel's classic compiler and compiler for E2K) can
// define __GNUC__ but the attribute must not be used with them.
// The new Clang-based ICX needs the attribute.
//
// NOTE: Build systems check for this too, keep them in sync with this.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__EDG__)
__attribute__((__target__("ssse3,sse4.1,pclmul")))
#endif
// The intrinsics use 16-byte-aligned reads from buf, thus they may read
// up to 15 bytes before or after the buffer (depending on the alignment
// of the buf argument). The values of the extra bytes are ignored.
// This unavoidably trips -fsanitize=address so address sanitizier has
// to be disabled for this function.
#if lzma_has_attribute(__no_sanitize_address__)
__attribute__((__no_sanitize_address__))
#endif
static uint64_t
crc64_clmul(const uint8_t *buf, size_t size, uint64_t crc)
{
	// The prototypes of the intrinsics use signed types while most of
	// the values are treated as unsigned here. These warnings in this
	// function have been checked and found to be harmless so silence them.
#if TUKLIB_GNUC_REQ(4, 6) || defined(__clang__)
#	pragma GCC diagnostic push
#	pragma GCC diagnostic ignored "-Wsign-conversion"
#	pragma GCC diagnostic ignored "-Wconversion"
#endif

#ifndef CRC_USE_GENERIC_FOR_SMALL_INPUTS
	// The code assumes that there is at least one byte of input.
	if (size == 0)
		return crc;
#endif

	// const uint64_t poly = 0xc96c5795d7870f42; // CRC polynomial
	const uint64_t p  = 0x92d8af2baf0e1e85; // (poly << 1) | 1
	const uint64_t mu = 0x9c3e466c172963d5; // (calc_lo(poly) << 1) | 1
	const uint64_t k2 = 0xdabe95afc7875f40; // calc_hi(poly, 1)
	const uint64_t k1 = 0xe05dd497ca393ae4; // calc_hi(poly, k2)
	const __m128i vfold0 = _mm_set_epi64x(p, mu);
	const __m128i vfold1 = _mm_set_epi64x(k2, k1);

	// Create a vector with 8-bit values 0 to 15. This is used to
	// construct control masks for _mm_blendv_epi8 and _mm_shuffle_epi8.
	const __m128i vramp = _mm_setr_epi32(
			0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c);

	// This is used to inverse the control mask of _mm_shuffle_epi8
	// so that bytes that wouldn't be picked with the original mask
	// will be picked and vice versa.
	const __m128i vsign = _mm_set1_epi8(0x80);

	// Memory addresses A to D and the distances between them:
	//
	//     A           B     C         D
	//     [skip_start][size][skip_end]
	//     [     size2      ]
	//
	// A and D are 16-byte aligned. B and C are 1-byte aligned.
	// skip_start and skip_end are 0-15 bytes. size is at least 1 byte.
	//
	// A = aligned_buf will initially point to this address.
	// B = The address pointed by the caller-supplied buf.
	// C = buf + size == aligned_buf + size2
	// D = buf + size + skip_end == aligned_buf + size2 + skip_end
	const size_t skip_start = (size_t)((uintptr_t)buf & 15);
	const size_t skip_end = (size_t)((0U - (uintptr_t)(buf + size)) & 15);
	const __m128i *aligned_buf = (const __m128i *)(
			(uintptr_t)buf & ~(uintptr_t)15);

	// If size2 <= 16 then the whole input fits into a single 16-byte
	// vector. If size2 > 16 then at least two 16-byte vectors must
	// be processed. If size2 > 16 && size <= 16 then there is only
	// one 16-byte vector's worth of input but it is unaligned in memory.
	//
	// NOTE: There is no integer overflow here if the arguments are valid.
	// If this overflowed, buf + size would too.
	size_t size2 = skip_start + size;

	// Masks to be used with _mm_blendv_epi8 and _mm_shuffle_epi8:
	// The first skip_start or skip_end bytes in the vectors will have
	// the high bit (0x80) set. _mm_blendv_epi8 and _mm_shuffle_epi8
	// will produce zeros for these positions. (Bitwise-xor of these
	// masks with vsign will produce the opposite behavior.)
	const __m128i mask_start
			= _mm_sub_epi8(vramp, _mm_set1_epi8(skip_start));
	const __m128i mask_end = _mm_sub_epi8(vramp, _mm_set1_epi8(skip_end));

	// Get the first 1-16 bytes into data0. If loading less than 16 bytes,
	// the bytes are loaded to the high bits of the vector and the least
	// significant positions are filled with zeros.
	const __m128i data0 = _mm_blendv_epi8(_mm_load_si128(aligned_buf),
			_mm_setzero_si128(), mask_start);
	++aligned_buf;

#if defined(__i386__) || defined(_M_IX86)
	const __m128i initial_crc = _mm_set_epi64x(0, ~crc);
#else
	// GCC and Clang would produce good code with _mm_set_epi64x
	// but MSVC needs _mm_cvtsi64_si128 on x86-64.
	const __m128i initial_crc = _mm_cvtsi64_si128(~crc);
#endif

	__m128i v0, v1, v2, v3;

#ifndef CRC_USE_GENERIC_FOR_SMALL_INPUTS
	if (size <= 16) {
		// Right-shift initial_crc by 1-16 bytes based on "size"
		// and store the result in v1 (high bytes) and v0 (low bytes).
		//
		// NOTE: The highest 8 bytes of initial_crc are zeros so
		// v1 will be filled with zeros if size >= 8. The highest 8
		// bytes of v1 will always become zeros.
		//
		// [      v1      ][      v0      ]
		//  [ initial_crc  ]                  size == 1
		//   [ initial_crc  ]                 size == 2
		//                [ initial_crc  ]    size == 15
		//                 [ initial_crc  ]   size == 16 (all in v0)
		const __m128i mask_low = _mm_add_epi8(
				vramp, _mm_set1_epi8(size - 16));
		MASK_LH(initial_crc, mask_low, v0, v1);

		if (size2 <= 16) {
			// There are 1-16 bytes of input and it is all
			// in data0. Copy the input bytes to v3. If there
			// are fewer than 16 bytes, the low bytes in v3
			// will be filled with zeros. That is, the input
			// bytes are stored to the same position as
			// (part of) initial_crc is in v0.
			MASK_L(data0, mask_end, v3);
		} else {
			// There are 2-16 bytes of input but not all bytes
			// are in data0.
			const __m128i data1 = _mm_load_si128(aligned_buf);

			// Collect the 2-16 input bytes from data0 and data1
			// to v2 and v3, and bitwise-xor them with the
			// low bits of initial_crc in v0. Note that the
			// the second xor is below this else-block as it
			// is shared with the other branch.
			MASK_H(data0, mask_end, v2);
			MASK_L(data1, mask_end, v3);
			v0 = _mm_xor_si128(v0, v2);
		}

		v0 = _mm_xor_si128(v0, v3);
		v1 = _mm_alignr_epi8(v1, v0, 8);
	} else
#endif
	{
		const __m128i data1 = _mm_load_si128(aligned_buf);
		MASK_LH(initial_crc, mask_start, v0, v1);
		v0 = _mm_xor_si128(v0, data0);
		v1 = _mm_xor_si128(v1, data1);

#define FOLD \
	v1 = _mm_xor_si128(v1, _mm_clmulepi64_si128(v0, vfold1, 0x00)); \
	v0 = _mm_xor_si128(v1, _mm_clmulepi64_si128(v0, vfold1, 0x11));

		while (size2 > 32) {
			++aligned_buf;
			size2 -= 16;
			FOLD
			v1 = _mm_load_si128(aligned_buf);
		}

		if (size2 < 32) {
			MASK_H(v0, mask_end, v2);
			MASK_L(v0, mask_end, v0);
			MASK_L(v1, mask_end, v3);
			v1 = _mm_or_si128(v2, v3);
		}

		FOLD
		v1 = _mm_srli_si128(v0, 8);
#undef FOLD
	}

	v1 = _mm_xor_si128(_mm_clmulepi64_si128(v0, vfold1, 0x10), v1);
	v0 = _mm_clmulepi64_si128(v1, vfold0, 0x00);
	v2 = _mm_clmulepi64_si128(v0, vfold0, 0x10);
	v0 = _mm_xor_si128(_mm_xor_si128(v2, _mm_slli_si128(v0, 8)), v1);

#if defined(__i386__) || defined(_M_IX86)
	return ~(((uint64_t)(uint32_t)_mm_extract_epi32(v0, 3) << 32) |
			(uint64_t)(uint32_t)_mm_extract_epi32(v0, 2));
#else
	return ~(uint64_t)_mm_extract_epi64(v0, 1);
#endif

#if TUKLIB_GNUC_REQ(4, 6) || defined(__clang__)
#	pragma GCC diagnostic pop
#endif
}
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER) && !defined(__clang__) \
		&& defined(_M_IX86)
#	pragma optimize("", on)
#endif
#endif


////////////////////////
// Detect CPU support //
////////////////////////

#if defined(CRC_GENERIC) && defined(CRC_CLMUL)
static inline bool
is_clmul_supported(void)
{
	int success = 1;
	uint32_t r[4]; // eax, ebx, ecx, edx

#if defined(_MSC_VER)
	// This needs <intrin.h> with MSVC. ICC has it as a built-in
	// on all platforms.
	__cpuid(r, 1);
#elif defined(HAVE_CPUID_H)
	// Compared to just using __asm__ to run CPUID, this also checks
	// that CPUID is supported and saves and restores ebx as that is
	// needed with GCC < 5 with position-independent code (PIC).
	success = __get_cpuid(1, &r[0], &r[1], &r[2], &r[3]);
#else
	// Just a fallback that shouldn't be needed.
	__asm__("cpuid\n\t"
			: "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3])
			: "a"(1), "c"(0));
#endif

	// Returns true if these are supported:
	// CLMUL (bit 1 in ecx)
	// SSSE3 (bit 9 in ecx)
	// SSE4.1 (bit 19 in ecx)
	const uint32_t ecx_mask = (1 << 1) | (1 << 9) | (1 << 19);
	return success && (r[2] & ecx_mask) == ecx_mask;

	// Alternative methods that weren't used:
	//   - ICC's _may_i_use_cpu_feature: the other methods should work too.
	//   - GCC >= 6 / Clang / ICX __builtin_cpu_supports("pclmul")
	//
	// CPUID decding is needed with MSVC anyway and older GCC. This keeps
	// the feature checks in the build system simpler too. The nice thing
	// about __builtin_cpu_supports would be that it generates very short
	// code as is it only reads a variable set at startup but a few bytes
	// doesn't matter here.
}


#ifdef HAVE_FUNC_ATTRIBUTE_CONSTRUCTOR
#	define CRC64_FUNC_INIT
#	define CRC64_SET_FUNC_ATTR __attribute__((__constructor__))
#else
#	define CRC64_FUNC_INIT = &crc64_dispatch
#	define CRC64_SET_FUNC_ATTR
static uint64_t crc64_dispatch(const uint8_t *buf, size_t size, uint64_t crc);
#endif


// Pointer to the the selected CRC64 method.
static uint64_t (*crc64_func)(const uint8_t *buf, size_t size, uint64_t crc)
		CRC64_FUNC_INIT;


CRC64_SET_FUNC_ATTR
static void
crc64_set_func(void)
{
	crc64_func = is_clmul_supported() ? &crc64_clmul : &crc64_generic;
	return;
}


#ifndef HAVE_FUNC_ATTRIBUTE_CONSTRUCTOR
static uint64_t
crc64_dispatch(const uint8_t *buf, size_t size, uint64_t crc)
{
	// When __attribute__((__constructor__)) isn't supported, set the
	// function pointer without any locking. If multiple threads run
	// the detection code in parallel, they will all end up setting
	// the pointer to the same value. This avoids the use of
	// mythread_once() on every call to lzma_crc64() but this likely
	// isn't strictly standards compliant. Let's change it if it breaks.
	crc64_set_func();
	return crc64_func(buf, size, crc);
}
#endif
#endif


extern LZMA_API(uint64_t)
lzma_crc64(const uint8_t *buf, size_t size, uint64_t crc)
{
#if defined(CRC_GENERIC) && defined(CRC_CLMUL)
	// If CLMUL is available, it is the best for non-tiny inputs,
	// being over twice as fast as the generic slice-by-four version.
	// However, for size <= 16 it's different. In the extreme case
	// of size == 1 the generic version can be five times faster.
	// At size >= 8 the CLMUL starts to become reasonable. It
	// varies depending on the alignment of buf too.
	//
	// The above doesn't include the overhead of mythread_once().
	// At least on x86-64 GNU/Linux, pthread_once() is very fast but
	// it still makes lzma_crc64(buf, 1, crc) 50-100 % slower. When
	// size reaches 12-16 bytes the overhead becomes negligible.
	//
	// So using the generic version for size <= 16 may give better
	// performance with tiny inputs but if such inputs happen rarely
	// it's not so obvious because then the lookup table of the
	// generic version may not be in the processor cache.
#ifdef CRC_USE_GENERIC_FOR_SMALL_INPUTS
	if (size <= 16)
		return crc64_generic(buf, size, crc);
#endif

/*
#ifndef HAVE_FUNC_ATTRIBUTE_CONSTRUCTOR
	// See crc64_dispatch(). This would be the alternative which uses
	// locking and doesn't use crc64_dispatch(). Note that on Windows
	// this method needs Vista threads.
	mythread_once(crc64_set_func);
#endif
*/

	return crc64_func(buf, size, crc);

#elif defined(CRC_CLMUL)
	// If CLMUL is used unconditionally without runtime CPU detection
	// then omitting the generic version and its 8 KiB lookup table
	// makes the library smaller.
	//
	// FIXME: Lookup table isn't currently omitted on 32-bit x86,
	// see crc64_table.c.
	return crc64_clmul(buf, size, crc);

#else
	return crc64_generic(buf, size, crc);
#endif
}
