///////////////////////////////////////////////////////////////////////////////
//
/// \file       common.h
/// \brief      Definitions common to the whole liblzma library
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef LZMA_COMMON_H
#define LZMA_COMMON_H

#include "sysdefs.h"
#include "mythread.h"
#include "tuklib_integer.h"

// LZMA_API_EXPORT is used to mark the exported API functions.
// It's used to define the LZMA_API macro.
//
// lzma_attr_visibility_hidden is used for marking *declarations* of extern
// variables that are internal to liblzma (-fvisibility=hidden alone is
// enough to hide the *definitions*). Such markings allow slightly more
// efficient code to accesses those variables in ELF shared libraries.
#if defined(_WIN32) || defined(__CYGWIN__)
#	ifdef DLL_EXPORT
#		define LZMA_API_EXPORT __declspec(dllexport)
#	else
#		define LZMA_API_EXPORT
#	endif
#	define lzma_attr_visibility_hidden
// Don't use ifdef or defined() below.
#elif HAVE_VISIBILITY
#	define LZMA_API_EXPORT __attribute__((__visibility__("default")))
#	define lzma_attr_visibility_hidden \
			__attribute__((__visibility__("hidden")))
#else
#	define LZMA_API_EXPORT
#	define lzma_attr_visibility_hidden
#endif

#define LZMA_API(type) LZMA_API_EXPORT type LZMA_API_CALL

#include "lzma.h"

// This is for detecting modern GCC and Clang attributes
// like __symver__ in GCC >= 10.
#ifdef __has_attribute
#	define lzma_has_attribute(attr) __has_attribute(attr)
#else
#	define lzma_has_attribute(attr) 0
#endif

// The extra symbol versioning in the C files may only be used when
// building a shared library. If HAVE_SYMBOL_VERSIONS_LINUX is defined
// to 2 then symbol versioning is done only if also PIC is defined.
// By default Libtool defines PIC when building a shared library and
// doesn't define it when building a static library but it can be
// overridden with --with-pic and --without-pic. configure let's rely
// on PIC if neither --with-pic or --without-pic was used.
#if defined(HAVE_SYMBOL_VERSIONS_LINUX) \
		&& (HAVE_SYMBOL_VERSIONS_LINUX == 2 && !defined(PIC))
#	undef HAVE_SYMBOL_VERSIONS_LINUX
#endif

#ifdef HAVE_SYMBOL_VERSIONS_LINUX
// To keep link-time optimization (LTO, -flto) working with GCC,
// the __symver__ attribute must be used instead of __asm__(".symver ...").
// Otherwise the symbol versions may be lost, resulting in broken liblzma
// that has wrong default versions in the exported symbol list!
// The attribute was added in GCC 10; LTO with older GCC is not supported.
//
// To keep -Wmissing-prototypes happy, use LZMA_SYMVER_API only with function
// declarations (including those with __alias__ attribute) and LZMA_API with
// the function definitions. This means a little bit of silly copy-and-paste
// between declarations and definitions though.
//
// As of GCC 12.2, the __symver__ attribute supports only @ and @@ but the
// very convenient @@@ isn't supported (it's supported by GNU assembler
// since 2000). When using @@ instead of @@@, the internal name must not be
// the same as the external name to avoid problems in some situations. This
// is why "#define foo_52 foo" is needed for the default symbol versions.
//
// __has_attribute is supported before GCC 10 and it is supported in Clang 14
// too (which doesn't support __symver__) so use it to detect if __symver__
// is available. This should be far more reliable than looking at compiler
// version macros as nowadays especially __GNUC__ is defined by many compilers.
#	if lzma_has_attribute(__symver__)
#		define LZMA_SYMVER_API(extnamever, type, intname) \
			extern __attribute__((__symver__(extnamever))) \
					LZMA_API(type) intname
#	else
#		define LZMA_SYMVER_API(extnamever, type, intname) \
			__asm__(".symver " #intname "," extnamever); \
			extern LZMA_API(type) intname
#	endif
#endif

// MSVC has __forceinline which shouldn't be combined with the inline keyword
// (results in a warning).
//
// GCC 3.1 added always_inline attribute so we don't need to check
// for __GNUC__ version. Similarly, all relevant Clang versions
// support it (at least Clang 3.0.0 does already).
// Other compilers might support too which also support __has_attribute
// (Solaris Studio) so do that check too.
#if defined(_MSC_VER)
#	define lzma_always_inline __forceinline
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER) \
		|| lzma_has_attribute(__always_inline__)
#	define lzma_always_inline inline __attribute__((__always_inline__))
#else
#	define lzma_always_inline inline
#endif

// These allow helping the compiler in some often-executed branches, whose
// result is almost always the same.
#ifdef __GNUC__
#	define likely(expr) __builtin_expect(expr, true)
#	define unlikely(expr) __builtin_expect(expr, false)
#else
#	define likely(expr) (expr)
#	define unlikely(expr) (expr)
#endif


/// Size of temporary buffers needed in some filters
#define LZMA_BUFFER_SIZE 4096


/// Maximum number of worker threads within one multithreaded component.
/// The limit exists solely to make it simpler to prevent integer overflows
/// when allocating structures etc. This should be big enough for now...
/// the code won't scale anywhere close to this number anyway.
#define LZMA_THREADS_MAX 16384


/// Starting value for memory usage estimates. Instead of calculating size
/// of _every_ structure and taking into account malloc() overhead etc., we
/// add a base size to all memory usage estimates. It's not very accurate
/// but should be easily good enough.
#define LZMA_MEMUSAGE_BASE (UINT64_C(1) << 15)

/// Start of internal Filter ID space. These IDs must never be used
/// in Streams.
#define LZMA_FILTER_RESERVED_START (LZMA_VLI_C(1) << 62)


/// Supported flags that can be passed to lzma_stream_decoder(),
/// lzma_auto_decoder(), or lzma_stream_decoder_mt().
#define LZMA_SUPPORTED_FLAGS \
	( LZMA_TELL_NO_CHECK \
	| LZMA_TELL_UNSUPPORTED_CHECK \
	| LZMA_TELL_ANY_CHECK \
	| LZMA_IGNORE_CHECK \
	| LZMA_CONCATENATED \
	| LZMA_FAIL_FAST )


/// Largest valid lzma_action value as unsigned integer.
#define LZMA_ACTION_MAX ((unsigned int)(LZMA_FULL_BARRIER))


/// Special return value (lzma_ret) to indicate that a timeout was reached
/// and lzma_code() must not return LZMA_BUF_ERROR. This is converted to
/// LZMA_OK in lzma_code().
#define LZMA_TIMED_OUT LZMA_RET_INTERNAL1

/// Special return value (lzma_ret) for use in stream_decoder_mt.c to
/// indicate Index was detected instead of a Block Header.
#define LZMA_INDEX_DETECTED LZMA_RET_INTERNAL2


typedef struct lzma_next_coder_s lzma_next_coder;

typedef struct lzma_filter_info_s lzma_filter_info;


/// Type of a function used to initialize a filter encoder or decoder
typedef lzma_ret (*lzma_init_function)(
		lzma_next_coder *next, const lzma_allocator *allocator,
		const lzma_filter_info *filters);

/// Type of a function to do some kind of coding work (filters, Stream,
/// Block encoders/decoders etc.). Some special coders use don't use both
/// input and output buffers, but for simplicity they still use this same
/// function prototype.
typedef lzma_ret (*lzma_code_function)(
		void *coder, const lzma_allocator *allocator,
		const uint8_t *restrict in, size_t *restrict in_pos,
		size_t in_size, uint8_t *restrict out,
		size_t *restrict out_pos, size_t out_size,
		lzma_action action);

/// Type of a function to free the memory allocated for the coder
typedef void (*lzma_end_function)(
		void *coder, const lzma_allocator *allocator);


/// Raw coder validates and converts an array of lzma_filter structures to
/// an array of lzma_filter_info structures. This array is used with
/// lzma_next_filter_init to initialize the filter chain.
struct lzma_filter_info_s {
	/// Filter ID. This can be used to share the same initiazation
	/// function *and* data structures with different Filter IDs
	/// (LZMA_FILTER_LZMA1EXT does it), and also by the encoder
	/// with lzma_filters_update() if filter chain is updated
	/// in the middle of a raw stream or Block (LZMA_SYNC_FLUSH).
	lzma_vli id;

	/// Pointer to function used to initialize the filter.
	/// This is NULL to indicate end of array.
	lzma_init_function init;

	/// Pointer to filter's options structure
	void *options;
};


/// Hold data and function pointers of the next filter in the chain.
struct lzma_next_coder_s {
	/// Pointer to coder-specific data
	void *coder;

	/// Filter ID. This is LZMA_VLI_UNKNOWN when this structure doesn't
	/// point to a filter coder.
	lzma_vli id;

	/// "Pointer" to init function. This is never called here.
	/// We need only to detect if we are initializing a coder
	/// that was allocated earlier. See lzma_next_coder_init and
	/// lzma_next_strm_init macros in this file.
	uintptr_t init;

	/// Pointer to function to do the actual coding
	lzma_code_function code;

	/// Pointer to function to free lzma_next_coder.coder. This can
	/// be NULL; in that case, lzma_free is called to free
	/// lzma_next_coder.coder.
	lzma_end_function end;

	/// Pointer to a function to get progress information. If this is NULL,
	/// lzma_stream.total_in and .total_out are used instead.
	void (*get_progress)(void *coder,
			uint64_t *progress_in, uint64_t *progress_out);

	/// Pointer to function to return the type of the integrity check.
	/// Most coders won't support this.
	lzma_check (*get_check)(const void *coder);

	/// Pointer to function to get and/or change the memory usage limit.
	/// If new_memlimit == 0, the limit is not changed.
	lzma_ret (*memconfig)(void *coder, uint64_t *memusage,
			uint64_t *old_memlimit, uint64_t new_memlimit);

	/// Update the filter-specific options or the whole filter chain
	/// in the encoder.
	lzma_ret (*update)(void *coder, const lzma_allocator *allocator,
			const lzma_filter *filters,
			const lzma_filter *reversed_filters);

	/// Set how many bytes of output this coder may produce at maximum.
	/// On success LZMA_OK must be returned.
	/// If the filter chain as a whole cannot support this feature,
	/// this must return LZMA_OPTIONS_ERROR.
	/// If no input has been given to the coder and the requested limit
	/// is too small, this must return LZMA_BUF_ERROR. If input has been
	/// seen, LZMA_OK is allowed too.
	lzma_ret (*set_out_limit)(void *coder, uint64_t *uncomp_size,
			uint64_t out_limit);
};


/// Macro to initialize lzma_next_coder structure
#define LZMA_NEXT_CODER_INIT \
	(lzma_next_coder){ \
		.coder = NULL, \
		.init = (uintptr_t)(NULL), \
		.id = LZMA_VLI_UNKNOWN, \
		.code = NULL, \
		.end = NULL, \
		.get_progress = NULL, \
		.get_check = NULL, \
		.memconfig = NULL, \
		.update = NULL, \
		.set_out_limit = NULL, \
	}


/// Internal data for lzma_strm_init, lzma_code, and lzma_end. A pointer to
/// this is stored in lzma_stream.
struct lzma_internal_s {
	/// The actual coder that should do something useful
	lzma_next_coder next;

	/// Track the state of the coder. This is used to validate arguments
	/// so that the actual coders can rely on e.g. that LZMA_SYNC_FLUSH
	/// is used on every call to lzma_code until next.code has returned
	/// LZMA_STREAM_END.
	enum {
		ISEQ_RUN,
		ISEQ_SYNC_FLUSH,
		ISEQ_FULL_FLUSH,
		ISEQ_FINISH,
		ISEQ_FULL_BARRIER,
		ISEQ_END,
		ISEQ_ERROR,
	} sequence;

	/// A copy of lzma_stream avail_in. This is used to verify that the
	/// amount of input doesn't change once e.g. LZMA_FINISH has been
	/// used.
	size_t avail_in;

	/// Indicates which lzma_action values are allowed by next.code.
	bool supported_actions[LZMA_ACTION_MAX + 1];

	/// If true, lzma_code will return LZMA_BUF_ERROR if no progress was
	/// made (no input consumed and no output produced by next.code).
	bool allow_buf_error;
};


/// Allocates memory
lzma_attr_alloc_size(1)
extern void *lzma_alloc(size_t size, const lzma_allocator *allocator);

/// Allocates memory and zeroes it (like calloc()). This can be faster
/// than lzma_alloc() + memzero() while being backward compatible with
/// custom allocators.
lzma_attr_alloc_size(1)
extern void *lzma_alloc_zero(size_t size, const lzma_allocator *allocator);

/// Frees memory
extern void lzma_free(void *ptr, const lzma_allocator *allocator);


/// Allocates strm->internal if it is NULL, and initializes *strm and
/// strm->internal. This function is only called via lzma_next_strm_init macro.
extern lzma_ret lzma_strm_init(lzma_stream *strm);

/// Initializes the next filter in the chain, if any. This takes care of
/// freeing the memory of previously initialized filter if it is different
/// than the filter being initialized now. This way the actual filter
/// initialization functions don't need to use lzma_next_coder_init macro.
extern lzma_ret lzma_next_filter_init(lzma_next_coder *next,
		const lzma_allocator *allocator,
		const lzma_filter_info *filters);

/// Update the next filter in the chain, if any. This checks that
/// the application is not trying to change the Filter IDs.
extern lzma_ret lzma_next_filter_update(
		lzma_next_coder *next, const lzma_allocator *allocator,
		const lzma_filter *reversed_filters);

/// Frees the memory allocated for next->coder either using next->end or,
/// if next->end is NULL, using lzma_free.
extern void lzma_next_end(lzma_next_coder *next,
		const lzma_allocator *allocator);


/// Copy as much data as possible from in[] to out[] and update *in_pos
/// and *out_pos accordingly. Returns the number of bytes copied.
extern size_t lzma_bufcpy(const uint8_t *restrict in, size_t *restrict in_pos,
		size_t in_size, uint8_t *restrict out,
		size_t *restrict out_pos, size_t out_size);


/// \brief      Return if expression doesn't evaluate to LZMA_OK
///
/// There are several situations where we want to return immediately
/// with the value of expr if it isn't LZMA_OK. This macro shortens
/// the code a little.
#define return_if_error(expr) \
do { \
	const lzma_ret ret_ = (expr); \
	if (ret_ != LZMA_OK) \
		return ret_; \
} while (0)


/// If next isn't already initialized, free the previous coder. Then mark
/// that next is _possibly_ initialized for the coder using this macro.
/// "Possibly" means that if e.g. allocation of next->coder fails, the
/// structure isn't actually initialized for this coder, but leaving
/// next->init to func is still OK.
#define lzma_next_coder_init(func, next, allocator) \
do { \
	if ((uintptr_t)(func) != (next)->init) \
		lzma_next_end(next, allocator); \
	(next)->init = (uintptr_t)(func); \
} while (0)


/// Initializes lzma_strm and calls func() to initialize strm->internal->next.
/// (The function being called will use lzma_next_coder_init()). If
/// initialization fails, memory that wasn't freed by func() is freed
/// along strm->internal.
#define lzma_next_strm_init(func, strm, ...) \
do { \
	return_if_error(lzma_strm_init(strm)); \
	const lzma_ret ret_ = func(&(strm)->internal->next, \
			(strm)->allocator, __VA_ARGS__); \
	if (ret_ != LZMA_OK) { \
		lzma_end(strm); \
		return ret_; \
	} \
} while (0)

#endif
