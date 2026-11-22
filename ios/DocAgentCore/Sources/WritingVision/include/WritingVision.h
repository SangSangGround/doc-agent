#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Gray byte buffers use an upright image with top-left origin. Corner/transform
// coordinates are normalized so capture and reference dimensions may differ.
typedef struct DAWritingTracker DAWritingTracker;
DAWritingTracker *DAWritingCreate(const uint8_t *gray, int width, int height, size_t stride,
                                  const double paperCorners[8]);
void DAWritingDestroy(DAWritingTracker *tracker);
bool DAWritingRegister(DAWritingTracker *tracker, const uint8_t *gray, int width, int height,
                       size_t stride, double referenceToLive[9]);

#ifdef __cplusplus
}
#endif
