///////////////////////////////////////////////////////////////////////////////
//
/// \file       lzip_decoder.h
/// \brief      Decodes .lz (lzip) files
//
//  Author:     Michał Górny
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef LZMA_LZIP_DECODER_H
#define LZMA_LZIP_DECODER_H

#include "common.h"

extern lzma_ret lzma_lzip_decoder_init(
		lzma_next_coder *next, const lzma_allocator *allocator,
		uint64_t memlimit, uint32_t flags);

#endif
