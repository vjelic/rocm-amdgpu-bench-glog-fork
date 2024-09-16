/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef __GFX_ROOFLINE_COMMON_H__
#define __GFX_ROOFLINE_COMMON_H__

#include <hip/hip_runtime.h>
#include <string>
#include <stdlib.h>
#include <cstdio>
#include <vector>
#include <cassert>

//#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#define HIP_ASSERT(x) { auto ret = (x); if(ret != hipSuccess) { fprintf(stderr, "%s:%d :: HIP error : %s\n", __FILE__, __LINE__, hipGetErrorString(ret));  throw std::runtime_error("hip_error"); }}

void help(void);
void showProgress(float percentage);

void stats(float *samples, int entries, float *mean, float *stdev, float *confidence);

std::string device_arch(int device_id);

static inline
void initHipEvents(hipEvent_t &start, hipEvent_t &stop)
{
    HIP_ASSERT(hipEventCreate(&start));
    HIP_ASSERT(hipEventCreate(&stop));
    HIP_ASSERT(hipEventRecord(start));
}


static inline
void stopHipEvents(float &eventMs, hipEvent_t &start, hipEvent_t &stop)
{
    HIP_ASSERT(hipEventRecord(stop));
    HIP_ASSERT(hipEventSynchronize(stop));
    HIP_ASSERT(hipEventElapsedTime(&eventMs, start, stop));
    HIP_ASSERT(hipEventDestroy(start));
    HIP_ASSERT(hipEventDestroy(stop));
}
#endif

