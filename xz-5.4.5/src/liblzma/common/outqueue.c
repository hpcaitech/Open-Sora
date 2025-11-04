///////////////////////////////////////////////////////////////////////////////
//
/// \file       outqueue.c
/// \brief      Output queue handling in multithreaded coding
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "outqueue.h"


/// Get the maximum number of buffers that may be allocated based
/// on the number of threads. For now this is twice the number of threads.
/// It's a compromise between RAM usage and keeping the worker threads busy
/// when buffers finish out of order.
#define GET_BUFS_LIMIT(threads) (2 * (threads))


extern uint64_t
lzma_outq_memusage(uint64_t buf_size_max, uint32_t threads)
{
	// This is to ease integer overflow checking: We may allocate up to
	// GET_BUFS_LIMIT(LZMA_THREADS_MAX) buffers and we need some extra
	// memory for other data structures too (that's the /2).
	//
	// lzma_outq_prealloc_buf() will still accept bigger buffers than this.
	const uint64_t limit
			= UINT64_MAX / GET_BUFS_LIMIT(LZMA_THREADS_MAX) / 2;

	if (threads > LZMA_THREADS_MAX || buf_size_max > limit)
		return UINT64_MAX;

	return GET_BUFS_LIMIT(threads)
			* lzma_outq_outbuf_memusage(buf_size_max);
}


static void
move_head_to_cache(lzma_outq *outq, const lzma_allocator *allocator)
{
	assert(outq->head != NULL);
	assert(outq->tail != NULL);
	assert(outq->bufs_in_use > 0);

	lzma_outbuf *buf = outq->head;
	outq->head = buf->next;
	if (outq->head == NULL)
		outq->tail = NULL;

	if (outq->cache != NULL && outq->cache->allocated != buf->allocated)
		lzma_outq_clear_cache(outq, allocator);

	buf->next = outq->cache;
	outq->cache = buf;

	--outq->bufs_in_use;
	outq->mem_in_use -= lzma_outq_outbuf_memusage(buf->allocated);

	return;
}


static void
free_one_cached_buffer(lzma_outq *outq, const lzma_allocator *allocator)
{
	assert(outq->cache != NULL);

	lzma_outbuf *buf = outq->cache;
	outq->cache = buf->next;

	--outq->bufs_allocated;
	outq->mem_allocated -= lzma_outq_outbuf_memusage(buf->allocated);

	lzma_free(buf, allocator);
	return;
}


extern void
lzma_outq_clear_cache(lzma_outq *outq, const lzma_allocator *allocator)
{
	while (outq->cache != NULL)
		free_one_cached_buffer(outq, allocator);

	return;
}


extern void
lzma_outq_clear_cache2(lzma_outq *outq, const lzma_allocator *allocator,
		size_t keep_size)
{
	if (outq->cache == NULL)
		return;

	// Free all but one.
	while (outq->cache->next != NULL)
		free_one_cached_buffer(outq, allocator);

	// Free the last one only if its size doesn't equal to keep_size.
	if (outq->cache->allocated != keep_size)
		free_one_cached_buffer(outq, allocator);

	return;
}


extern lzma_ret
lzma_outq_init(lzma_outq *outq, const lzma_allocator *allocator,
		uint32_t threads)
{
	if (threads > LZMA_THREADS_MAX)
		return LZMA_OPTIONS_ERROR;

	const uint32_t bufs_limit = GET_BUFS_LIMIT(threads);

	// Clear head/tail.
	while (outq->head != NULL)
		move_head_to_cache(outq, allocator);

	// If new buf_limit is lower than the old one, we may need to free
	// a few cached buffers.
	while (bufs_limit < outq->bufs_allocated)
		free_one_cached_buffer(outq, allocator);

	outq->bufs_limit = bufs_limit;
	outq->read_pos = 0;

	return LZMA_OK;
}


extern void
lzma_outq_end(lzma_outq *outq, const lzma_allocator *allocator)
{
	while (outq->head != NULL)
		move_head_to_cache(outq, allocator);

	lzma_outq_clear_cache(outq, allocator);
	return;
}


extern lzma_ret
lzma_outq_prealloc_buf(lzma_outq *outq, const lzma_allocator *allocator,
		size_t size)
{
	// Caller must have checked it with lzma_outq_has_buf().
	assert(outq->bufs_in_use < outq->bufs_limit);

	// If there already is appropriately-sized buffer in the cache,
	// we need to do nothing.
	if (outq->cache != NULL && outq->cache->allocated == size)
		return LZMA_OK;

	if (size > SIZE_MAX - sizeof(lzma_outbuf))
		return LZMA_MEM_ERROR;

	const size_t alloc_size = lzma_outq_outbuf_memusage(size);

	// The cache may have buffers but their size is wrong.
	lzma_outq_clear_cache(outq, allocator);

	outq->cache = lzma_alloc(alloc_size, allocator);
	if (outq->cache == NULL)
		return LZMA_MEM_ERROR;

	outq->cache->next = NULL;
	outq->cache->allocated = size;

	++outq->bufs_allocated;
	outq->mem_allocated += alloc_size;

	return LZMA_OK;
}


extern lzma_outbuf *
lzma_outq_get_buf(lzma_outq *outq, void *worker)
{
	// Caller must have used lzma_outq_prealloc_buf() to ensure these.
	assert(outq->bufs_in_use < outq->bufs_limit);
	assert(outq->bufs_in_use < outq->bufs_allocated);
	assert(outq->cache != NULL);

	lzma_outbuf *buf = outq->cache;
	outq->cache = buf->next;
	buf->next = NULL;

	if (outq->tail != NULL) {
		assert(outq->head != NULL);
		outq->tail->next = buf;
	} else {
		assert(outq->head == NULL);
		outq->head = buf;
	}

	outq->tail = buf;

	buf->worker = worker;
	buf->finished = false;
	buf->finish_ret = LZMA_STREAM_END;
	buf->pos = 0;
	buf->decoder_in_pos = 0;

	buf->unpadded_size = 0;
	buf->uncompressed_size = 0;

	++outq->bufs_in_use;
	outq->mem_in_use += lzma_outq_outbuf_memusage(buf->allocated);

	return buf;
}


extern bool
lzma_outq_is_readable(const lzma_outq *outq)
{
	if (outq->head == NULL)
		return false;

	return outq->read_pos < outq->head->pos || outq->head->finished;
}


extern lzma_ret
lzma_outq_read(lzma_outq *restrict outq,
		const lzma_allocator *restrict allocator,
		uint8_t *restrict out, size_t *restrict out_pos,
		size_t out_size,
		lzma_vli *restrict unpadded_size,
		lzma_vli *restrict uncompressed_size)
{
	// There must be at least one buffer from which to read.
	if (outq->bufs_in_use == 0)
		return LZMA_OK;

	// Get the buffer.
	lzma_outbuf *buf = outq->head;

	// Copy from the buffer to output.
	//
	// FIXME? In threaded decoder it may be bad to do this copy while
	// the mutex is being held.
	lzma_bufcpy(buf->buf, &outq->read_pos, buf->pos,
			out, out_pos, out_size);

	// Return if we didn't get all the data from the buffer.
	if (!buf->finished || outq->read_pos < buf->pos)
		return LZMA_OK;

	// The buffer was finished. Tell the caller its size information.
	if (unpadded_size != NULL)
		*unpadded_size = buf->unpadded_size;

	if (uncompressed_size != NULL)
		*uncompressed_size = buf->uncompressed_size;

	// Remember the return value.
	const lzma_ret finish_ret = buf->finish_ret;

	// Free this buffer for further use.
	move_head_to_cache(outq, allocator);
	outq->read_pos = 0;

	return finish_ret;
}


extern void
lzma_outq_enable_partial_output(lzma_outq *outq,
		void (*enable_partial_output)(void *worker))
{
	if (outq->head != NULL && !outq->head->finished
			&& outq->head->worker != NULL) {
		enable_partial_output(outq->head->worker);

		// Set it to NULL since calling it twice is pointless.
		outq->head->worker = NULL;
	}

	return;
}
