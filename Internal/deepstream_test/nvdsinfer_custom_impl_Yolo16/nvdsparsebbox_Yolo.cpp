/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"

static const int NUM_CLASSES_YOLO = 16;

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

static inline float clampVal(float val, float minVal, float maxVal)
{
    return std::max(minVal, std::min(maxVal, val));
}

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV7(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

/* YOLOv4 implementations */
static NvDsInferParseObjectInfo convertBBoxYoloV4(const float& bx1, const float& by1, const float& bx2,
                                     const float& by2, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution

    float x1 = bx1 * netW;
    float y1 = by1 * netH;
    float x2 = bx2 * netW;
    float y2 = by2 * netH;

    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);
    x2 = clamp(x2, 0, netW);
    y2 = clamp(y2, 0, netH);

    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH);

    return b;
}

static void addBBoxProposalYoloV4(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxYoloV4(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV4Tensor(
    const float* boxes, const float* scores, const float* classes,
    const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    const uint& netW, const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;

    uint bbox_location = 0;
    for (uint b = 0; b < num_bboxes; ++b)
    {
        float bx1 = boxes[bbox_location];
        float by1 = boxes[bbox_location + 1];
        float bx2 = boxes[bbox_location + 2];
        float by2 = boxes[bbox_location + 3];

	int class_id = (int)classes[b];

        if (class_id < detectionParams.numClassesConfigured && class_id >=0)
        {
            float prob = scores[b];
            if (prob > detectionParams.perClassPreclusterThreshold[class_id])
            {
                addBBoxProposalYoloV4(bx1, by1, bx2, by2, netW, netH, class_id, prob, binfo);
            }

        }


        bbox_location += 4;
    }

    return binfo;
}

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    const NvDsInferLayerInfo *boxes = NULL;
    const NvDsInferLayerInfo *scores = NULL;
    const NvDsInferLayerInfo *num = NULL;
    const NvDsInferLayerInfo *classes_layer = NULL;
    int outputLayernum = outputLayersInfo.size();
    for(int l=0; l<outputLayernum; l++) {
	if (!strcmp(outputLayersInfo[l].layerName, "num_detections")) {
            num = &outputLayersInfo[l];
	}
        if (!strcmp(outputLayersInfo[l].layerName, "nmsed_boxes")) {
            boxes = &outputLayersInfo[l];
	}
	if (!strcmp(outputLayersInfo[l].layerName, "nmsed_scores")) {
            scores = &outputLayersInfo[l];
	}
	if (!strcmp(outputLayersInfo[l].layerName, "nmsed_classes")) {
            classes_layer = &outputLayersInfo[l];
	}
    }
    int * num_layer = (int *)num->buffer;

    std::vector<NvDsInferParseObjectInfo> objects;

    uint num_bboxes = num_layer[0];
    // 2 dimensional: [num_bboxes, 4]
    assert(boxes->inferDims.numDims == 2);
    // Single dimensional: [num_bboxes]
    assert(scores->inferDims.numDims == 1);

    // std::cout << "Network Info: " << networkInfo.height << "  " << networkInfo.width << std::endl;

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeYoloV4Tensor(
            (const float*)(boxes->buffer), (const float*)(scores->buffer), (const float*)(classes_layer->buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}
/* YOLOv4 implementations end*/

/*Yolov7 bbox parser*/
static NvDsInferParseObjectInfo convertBBoxYoloV7(const float& bx, const float& by, const float& bw,
                                     const float& bh, const int& stride, const uint& netW,
                                     const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution
    float xCenter = bx * stride;
    float yCenter = by * stride;
    float x0 = xCenter - bw / 2;
    float y0 = yCenter - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;

    x0 = clamp(x0, 0, netW);
    y0 = clamp(y0, 0, netH);
    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);

    b.left = x0;
    b.width = clamp(x1 - x0, 0, netW);
    b.top = y0;
    b.height = clamp(y1 - y0, 0, netH);

    return b;
}

static void addBBoxProposalYoloV7(const float bx, const float by, const float bw, const float bh,
                     const uint stride, const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxYoloV7(bx, by, bw, bh, stride, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static bool NvDsInferParseYoloV7(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
 

    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer in bbox parsing" << std::endl;;
        return false;
    }
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<NvDsInferParseObjectInfo> objects;

    float* data = (float*)layer.buffer;
    const int dimensions = layer.inferDims.d[1];
    int rows = layer.inferDims.numElements / layer.inferDims.d[1];

    for (int i = 0; i < rows; ++i) {
        //85 = x, y, w, h, maxProb, score0......score79
        float bx = data[ 0];
        float by = data[ 1];
        float bw = data[ 2];
        float bh = data[ 3];
        float maxProb = data[ 4];
        int  maxIndex;
        float * classes_scores = data + 5;
        
        float maxScore = 0;
        int index = 0;
        for (int j = 0 ;j < NUM_CLASSES_YOLO; j++){
           if(*classes_scores > maxScore){
              index = j;
              maxScore = *classes_scores;
           }
           classes_scores++;
        }

        maxIndex = index;
        data += dimensions;
        
        addBBoxProposalYoloV7(bx, by, bw, bh, 1, networkInfo.width, networkInfo.height, maxIndex, maxProb, objects);    
    }
    objectList = objects;
    return true;
}

extern "C" bool NvDsInferParseCustomYoloV7(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV7 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}


static bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer in bbox parsing" << std::endl;;
        return false;
    }
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<NvDsInferParseObjectInfo> objects;

    float* data = (float*)layer.buffer;
    const int dimensions = layer.inferDims.d[1];
    int rows = layer.inferDims.numElements / layer.inferDims.d[1];


    for (int i = 0; i < rows; ++i) {
        //85 = x, y, w, h, score0......score79
        float bx = data[ 0];
        float by = data[ 1];
        float bw = data[ 2];
        float bh = data[ 3];
        float * classes_scores = data + 4;

        float maxScore = 0;
        int index = 0;
        for (int j = 0 ;j < NUM_CLASSES_YOLO; j++){
           if(*classes_scores > maxScore){
              index = j;
              maxScore = *classes_scores;
           }
           classes_scores++;
        }

        int maxIndex = index;
        data += dimensions;
        float maxProb = 1.0;
        // share the same addBBoxProposal function
        addBBoxProposalYoloV7(bx, by, bw, bh, 1, networkInfo.width, networkInfo.height, maxIndex, maxScore, objects);

    }
    objectList = objects;
    return true;
}

extern "C" bool NvDsInferParseCustomYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV8 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

static bool
decodeYoloV11Tensor(
    const float* data, int numFeatures, int numAnchors, int numClasses,
    const NvDsInferNetworkInfo& networkInfo,
    const NvDsInferParseDetectionParams& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    const uint netW = networkInfo.width;
    const uint netH = networkInfo.height;

    for (int anchor = 0; anchor < numAnchors; ++anchor) {
        const float cx = data[0 * numAnchors + anchor];
        const float cy = data[1 * numAnchors + anchor];
        const float w = data[2 * numAnchors + anchor];
        const float h = data[3 * numAnchors + anchor];

        float maxProb = 0.0f;
        int maxIndex = 0;
        for (int c = 0; c < numClasses; ++c) {
            const float prob = data[(4 + c) * numAnchors + anchor];
            if (prob > maxProb) {
                maxProb = prob;
                maxIndex = c;
            }
        }

        if (maxProb < detectionParams.perClassPreclusterThreshold[maxIndex]) {
            continue;
        }

        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        x1 = clampVal(x1, 0.0f, static_cast<float>(netW));
        y1 = clampVal(y1, 0.0f, static_cast<float>(netH));
        x2 = clampVal(x2, 0.0f, static_cast<float>(netW));
        y2 = clampVal(y2, 0.0f, static_cast<float>(netH));

        const float width = x2 - x1;
        const float height = y2 - y1;
        if (width < 1.0f || height < 1.0f) {
            continue;
        }

        NvDsInferObjectDetectionInfo obj{};
        obj.classId = maxIndex;
        obj.detectionConfidence = maxProb;
        obj.left = x1;
        obj.top = y1;
        obj.width = width;
        obj.height = height;
        obj.rotation_angle = 0.0f;

        objectList.push_back(obj);
    }

    return true;
}

static bool
NvDsInferParseCustomYoloV11(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: No output layers found for YOLOv11 parsing" << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& output = outputLayersInfo[0];
    const float* data = static_cast<const float*>(output.buffer);

    if (output.inferDims.numDims < 2) {
        std::cerr << "ERROR: Unexpected YOLOv11 output dimensions" << std::endl;
        return false;
    }

    const int numFeatures = output.inferDims.d[0];
    const int numAnchors = output.inferDims.d[1];
    // YOLO11-seg models pack 4 box + numClassesConfigured class scores +
    // 32 mask coefficients into each anchor's feature vector (e.g. 52 = 4 +
    // 16 + 32 for this 16-class model).  Only the first
    // detectionParams.numClassesConfigured channels after the box are real
    // class scores -- the rest are mask coefficients, not bounded like a
    // probability, so they must never enter the class arg-max or they win it
    // and corrupt/overrun the per-class threshold lookup.  Boxes only here;
    // mask decoding is not implemented by this parser.
    const int numClasses = static_cast<int>(detectionParams.numClassesConfigured);

    if (numClasses <= 0 || numFeatures < 4 + numClasses) {
        std::cerr << "ERROR: Invalid YOLOv11 feature count: " << numFeatures
                   << " for " << numClasses << " configured classes" << std::endl;
        return false;
    }

    objectList.clear();
    objectList.reserve(numAnchors);

    return decodeYoloV11Tensor(
        data, numFeatures, numAnchors, numClasses, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParseYoloV11(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    return NvDsInferParseCustomYoloV11(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

// ---------------------------------------------------------------------------
// YOLO11-seg instance-mask parser.
//
// output0: [1, 4 + numClasses + 32, numAnchors]  -- box, class scores, then
//          32 mask coefficients per anchor (Ultralytics YOLO-seg layout).
// output1: [1, 32, protoH, protoW]                -- mask prototypes.
//
// A per-object mask is sigmoid(coeffs . prototypes), cropped to the box in
// *prototype-grid* coordinates and left at that native resolution -- nvdsosd
// stretches a bbox-local mask over the object's rect_params at render time,
// so no manual upsampling to box-pixel size is needed here.
//
// nvinfer's own NMS/clustering does not run on this path (network-type=3),
// so this parser also does its own per-class greedy IoU suppression.
// ---------------------------------------------------------------------------

struct SegCandidate {
    int classId;
    float conf;
    float x1, y1, x2, y2;  // net-pixel space
    const float* coeffs;   // pointer into output0, 32 values, stride numAnchors apart
    int anchor;
};

static float box_iou(const SegCandidate& a, const SegCandidate& b)
{
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float areaA = std::max(1e-6f, (a.x2 - a.x1) * (a.y2 - a.y1));
    float areaB = std::max(1e-6f, (b.x2 - b.x1) * (b.y2 - b.y1));
    return inter / (areaA + areaB - inter);
}

extern "C" bool
NvDsInferParseYoloV11Seg(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
    objectList.clear();

    const NvDsInferLayerInfo* det = nullptr;
    const NvDsInferLayerInfo* proto = nullptr;
    for (auto const& layer : outputLayersInfo) {
        if (layer.inferDims.numDims == 2) det = &layer;
        else if (layer.inferDims.numDims == 3) proto = &layer;
    }
    if (!det || !proto) {
        std::cerr << "ERROR: YOLOv11-seg expects a [features,anchors] detection "
                     "output and a [maskCoeffs,protoH,protoW] prototype output"
                  << std::endl;
        return false;
    }

    const int numFeatures = det->inferDims.d[0];
    const int numAnchors = det->inferDims.d[1];
    const int numClasses = static_cast<int>(detectionParams.numClassesConfigured);
    const int maskCoeffs = proto->inferDims.d[0];
    const int protoH = proto->inferDims.d[1];
    const int protoW = proto->inferDims.d[2];

    if (numClasses <= 0 || numFeatures < 4 + numClasses + maskCoeffs) {
        std::cerr << "ERROR: YOLOv11-seg feature layout mismatch: numFeatures="
                  << numFeatures << " numClasses=" << numClasses
                  << " maskCoeffs=" << maskCoeffs << std::endl;
        return false;
    }

    const float* data = static_cast<const float*>(det->buffer);
    const float* protoData = static_cast<const float*>(proto->buffer);
    const uint netW = networkInfo.width;
    const uint netH = networkInfo.height;
    const int coeffBase = 4 + numClasses;  // first mask-coefficient row

    // 1. threshold -> candidates (box, class, confidence, pointer to this
    //    anchor's mask coefficients -- still strided through `data`).
    std::vector<SegCandidate> candidates;
    candidates.reserve(256);
    for (int anchor = 0; anchor < numAnchors; ++anchor) {
        float maxProb = 0.0f;
        int maxIndex = 0;
        for (int c = 0; c < numClasses; ++c) {
            float prob = data[(4 + c) * numAnchors + anchor];
            if (prob > maxProb) { maxProb = prob; maxIndex = c; }
        }
        if (maxProb < detectionParams.perClassPreclusterThreshold[maxIndex]) continue;

        float cx = data[0 * numAnchors + anchor];
        float cy = data[1 * numAnchors + anchor];
        float w = data[2 * numAnchors + anchor];
        float h = data[3 * numAnchors + anchor];
        float x1 = clampVal(cx - w * 0.5f, 0.0f, (float)netW);
        float y1 = clampVal(cy - h * 0.5f, 0.0f, (float)netH);
        float x2 = clampVal(cx + w * 0.5f, 0.0f, (float)netW);
        float y2 = clampVal(cy + h * 0.5f, 0.0f, (float)netH);
        if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) continue;

        SegCandidate sc;
        sc.classId = maxIndex; sc.conf = maxProb;
        sc.x1 = x1; sc.y1 = y1; sc.x2 = x2; sc.y2 = y2;
        sc.coeffs = data + (size_t)coeffBase * numAnchors;  // + anchor, per-row, below
        sc.anchor = anchor;
        candidates.push_back(sc);
    }

    // cap candidate count before O(n^2) NMS
    std::sort(candidates.begin(), candidates.end(),
              [](const SegCandidate& a, const SegCandidate& b) { return a.conf > b.conf; });
    const size_t topk = 300;
    if (candidates.size() > topk) candidates.resize(topk);

    // 2. per-class greedy NMS
    std::vector<bool> suppressed(candidates.size(), false);
    const float nmsIou = 0.45f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            if (candidates[j].classId != candidates[i].classId) continue;
            if (box_iou(candidates[i], candidates[j]) > nmsIou) suppressed[j] = true;
        }
    }

    // 3. decode a bbox-local mask for every surviving candidate
    const float scaleX = (float)protoW / (float)netW;
    const float scaleY = (float)protoH / (float)netH;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        const SegCandidate& sc = candidates[i];

        int px1 = clampVal(std::floor(sc.x1 * scaleX), 0.0f, (float)protoW);
        int py1 = clampVal(std::floor(sc.y1 * scaleY), 0.0f, (float)protoH);
        int px2 = clampVal(std::ceil(sc.x2 * scaleX), 0.0f, (float)protoW);
        int py2 = clampVal(std::ceil(sc.y2 * scaleY), 0.0f, (float)protoH);
        int mw = std::max(1, px2 - px1);
        int mh = std::max(1, py2 - py1);

        float* mask = new float[mw * mh];
        for (int yy = 0; yy < mh; ++yy) {
            for (int xx = 0; xx < mw; ++xx) {
                int py = py1 + yy, pxg = px1 + xx;
                float acc = 0.0f;
                for (int c = 0; c < maskCoeffs; ++c) {
                    float coeff = sc.coeffs[(size_t)c * numAnchors + sc.anchor];
                    float protoVal = protoData[((size_t)c * protoH + py) * protoW + pxg];
                    acc += coeff * protoVal;
                }
                mask[yy * mw + xx] = 1.0f / (1.0f + std::exp(-acc));  // sigmoid
            }
        }

        NvDsInferInstanceMaskInfo obj{};
        obj.classId = sc.classId;
        obj.detectionConfidence = sc.conf;
        obj.left = sc.x1;
        obj.top = sc.y1;
        obj.width = sc.x2 - sc.x1;
        obj.height = sc.y2 - sc.y1;
        obj.mask = mask;
        obj.mask_width = mw;
        obj.mask_height = mh;
        obj.mask_size = mw * mh * sizeof(float);
        objectList.push_back(obj);
    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV7);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV8);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV11);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV11Seg);
