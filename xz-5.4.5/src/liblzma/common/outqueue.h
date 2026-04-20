///////////////////////////////////////////////////////////////////////////////
//
/// \file       outqueue.h
/// \brief      Output queue handling in multithreaded coding
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "common.h"


/// Output buffer for a single thread
typedef struct lzma_outbuf_s lzma_outbuf;
struct lzma_outbuf_s {
	/// Pointer to the next buffer. This is used for the cached buffers.
	/// The worker thread must not modify this.
	lzma_outbuf *next;

	/// This initialized by lzma_outq_get_buf() and
	/// is used by lzma_outq_enable_partial_output().
	/// The worker thread must not modify this.
	void *worker;

	/// Amount of memory allocated for buf[].
	/// The worker thread must not modify this.
	size_t allocated;

	/// Writing position in the worker thread or, in other words, the
	/// amount of finished data written to buf[] which can be copied out
	///
	/// \note       This is read by another thread and thus access
	///             to this variable needs a mutex.
	size_t pos;

	/// Decompression: Position in the input buffer in the worker thread
	/// that matches the output "pos" above. This is used to detect if
	/// more output might be possible from the worker thread: if it has
	/// consumed all its input, then more output isn't possible.
	///
	/// \note       This is read by another thread and thus access
	///             to this variable needs a mutex.
	size_t decoder_in_pos;

	/// True when no more data will be written into this buffer.
	///
	/// \note       This is read by another thread and thus access
	///             to this variable needs a mutex.
	bool finished;

	/// Return value for lzma_outq_read() when the last byte from
	/// a finished buffer has been read. Defaults to LZMA_STREAM_END.
	/// This must *not* be LZMA_OK. The idea is to allow a decoder to
	/// pass an error code to the main thread, setting the code here
	/// together with finished = true.
	lzma_ret finish_ret;

	/// Additional size information. lzma_outq_read() may read these
	/// when "finished" is true.
	lzma_vli unpadded_size;
	lzma_vli uncompressed_size;

	/// Buffer of "allocated" bytes
	uint8_t buf[];
};


typedef struct {
	/// Linked list of buffers in use. The next output byte will be
	/// read from the head and buffers for the next thread will be
	/// appended to the tail. tail->next is always NULL.
	lzma_outbuf *head;
	lzma_outbuf *tail;

	/// Number of bytes read from head->buf[] in lzma_outq_read()
	size_t read_pos;

	/// Linked list of allocated buffers that aren't currently used.
	/// This way buffers of similar size can be reused and don't
	/// need to be reallocated every time. For simplicity, all
	/// cached buffers in the list have the same allocated size.
	lzma_outbuf *cache;

	/// Total amount of memory allocated for buffers
	uint64_t mem_allocated;

	/// Amount of memory used by the buffers that are in use in
	/// the head...tail linked list.
	uint64_t mem_in_use;

	/// Number of buffers in use in the head...tail list. If and only if
	/// this is zero, the pointers head and tail above are NULL.
	uint32_t bufs_in_use;

	/// Number of buffers allocated (in use + cached)
	uint32_t bufs_allocated;

	/// Maximum allowed number of allocated buffers
	uint32_t bufs_limit;
} lzma_outq;


/**
 * \brief       Calculate the memory usage of an output queue
 *
 * \return      Approximate memory usage in bytes or UINT64_MAX on error.
 */
extern uint64_t lzma_outq_memusage(uint64_t buf_size_max, uint32_t threads);


/// \brief      Initialize an output queue
///
/// \param      outq            Pointer to an output queue. Before calling
///                             this function the first time, *outq should
///                             have been zeroed with memzero() so that this
///                             function knows that there are no previous
///                             allocations to free.
/// \param      allocator       Pointer to allocator or NULL
/// \param      threads         Number of buffers that may be in use
///                             concurrently. Note that more than this number
///                             of buffers may actually get allocated to
///                             improve performance when buffers finish
///                             out of order. The actual maximum number of
///                             allocated buffers is derived from the number
///                             of threads.
///
/// \return     - LZMA_OK
///             - LZMA_MEM_ERROR
///
extern lzma_ret lzma_outq_init(lzma_outq *outq,
		const lzma_allocator *allocator, uint32_t threads);


/// \brief      Free the memory associated with the output queue
extern void lzma_outq_end(lzma_outq *outq, const lzma_allocator *allocator);


/// \brief      Free all cached buffers that consume memory but aren't in use
extern void lzma_outq_clear_cache(
		lzma_outq *outq, const lzma_allocator *allocator);


/// \brief      Like lzma_outq_clear_cache() but might keep one buffer
///
/// One buffer is not freed if its size is equal to keep_size.
/// This is useful if the caller knows that it will soon need a buffer of
/// keep_size bytes. This way it won't be freed and immediately reallocated.
extern void lzma_outq_clear_cache2(
		lzma_outq *outq, const lzma_allocator *allocator,
		size_t keep_size);


/// \brief      Preallocate a new buffer into cache
///
/// Splitting the buffer allocation into a separate function makes it
/// possible to ensure that way lzma_outq_get_buf() cannot fail.
/// If the preallocated buffer isn't actually used (for example, some
/// other error occurs), the caller has to do nothing as the buffer will
/// be used later or cleared from the cache when not needed.
///
/// \return     LZMA_OK on success, LZMA_MEM_ERROR if allocation fails
///
extern lzma_ret lzma_outq_prealloc_buf(
		lzma_outq *outq, const lzma_allocator *allocator, size_t size);


/// \brief      Get a new buffer
///
/// lzma_outq_prealloc_buf() must be used to ensure that there is a buffer
/// available before calling lzma_outq_get_buf().
///
extern lzma_outbuf *lzma_outq_get_buf(lzma_outq *outq, void *worker);


/// \brief      Test if there is data ready to be read
///
/// Call to this function must be protected with the same mutex that
/// is used to protect lzma_outbuf.finished.
///
extern bool lzma_outq_is_readable(const lzma_outq *outq);


/// \brief      Read finished data
///
/// \param      outq            Pointer to an output queue
/// \param      out             Beginning of the output buffer
/// \param      out_pos         The next byte will be written to
///                             out[*out_pos].
/// \param      out_size        Size of the out buffer; the first byte into
///                             which no data is written to is out[out_size].
/// \param      unpadded_size   Unpadded Size from the Block encoder
/// \param      uncompressed_size Uncompressed Size from the Block encoder
///
/// \return     - LZMA: All OK. Either no data was available or the buffer
///               being read didn't become empty yet.
///             - LZMA_STREAM_END: The buffer being read was finished.
///               *unpadded_size and *uncompressed_size were set if they
///               were not NULL.
///
/// \note       This reads lzma_outbuf.finished and .pos variables and thus
///             calls to this function need to be protected with a mutex.
///
extern lzma_ret lzma_outq_read(lzma_outq *restrict outq,
		const lzma_allocator *restrict allocator,
		uint8_t *restrict out, size_t *restrict out_pos,
		size_t out_size, lzma_vli *restrict unpadded_size,
		lzma_vli *restrict uncompressed_size);


/// \brief      Enable partial output from a worker thread
///
/// If the buffer at the head of the output queue isn't finished,
/// this will call enable_partial_output on the worker associated with
/// that output buffer.
///
/// \note       This reads a lzma_outbuf.finished variable and thus
///             calls to this function need to be protected with a mutex.
///
extern void lzma_outq_enable_partial_output(lzma_outq *outq,
		void (*enable_partial_output)(void *worker));


/// \brief      Test if there is at least one buffer free
///
/// This must be used before getting a new buffer with lzma_outq_get_buf().
///
static inline bool
lzma_outq_has_buf(const lzma_outq *outq)
{
	return outq->bufs_in_use < outq->bufs_limit;
}


/// \brief      Test if the queue is completely empty
static inline bool
lzma_outq_is_empty(const lzma_outq *outq)
{
	return outq->bufs_in_use == 0;
}


/// \brief      Get the amount of memory needed for a single lzma_outbuf
///
/// \note       Caller must check that the argument is significantly less
///             than SIZE_MAX to avoid an integer overflow!
static inline uint64_t
lzma_outq_outbuf_memusage(size_t buf_size)
{
	assert(buf_size <= SIZE_MAX - sizeof(lzma_outbuf));
	return sizeof(lzma_outbuf) + buf_size;
}
