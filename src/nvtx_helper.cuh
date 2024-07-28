/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
 #ifndef __NVTX_HELPER_CUH
 #define __NVTX_HELPER_CUH
 
 #ifdef USE_NVTX
 #include <nvToolsExt.h>
 
 static const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
 static const int num_colors = sizeof(colors)/sizeof(uint32_t);
 
 #define PUSH_RANGE(name,cid) { \
     int color_id = cid; \
     color_id = color_id%num_colors;\
     nvtxEventAttributes_t eventAttrib = {0}; \
     eventAttrib.version = NVTX_VERSION; \
     eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
     eventAttrib.colorType = NVTX_COLOR_ARGB; \
     eventAttrib.color = colors[color_id]; \
     eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
     eventAttrib.message.ascii = name; \
     nvtxRangePushEx(&eventAttrib); \
 }
 #define POP_RANGE nvtxRangePop();
 #else
 #define PUSH_RANGE(name,cid)
 #define POP_RANGE
 #endif
 
 #endif //NVTX_HELPER_CUH
 