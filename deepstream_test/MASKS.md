# Pixel-level masks in DeepStream

Yes — the same instance masks the `.pt` model gives you. The model already
emits them and the parser already decodes them; two things were flattening
them at render time.

## What was actually wrong

Measured on frame 45000 of the camA hour, decoding `model.onnx` exactly the way
`NvDsInferParseYoloV11Seg` does:

| class | box (net px) | mask cells | cells > 0.0 | cells > 0.5 |
|---|---|---|---|---|
| yellow_mustard_sauce | 14×33 | 5×10 | **100.0 %** | 56.0 % |
| scoop | 34×24 | 10×7 | **100.0 %** | 45.7 % |
| ketchup_sauce | 13×28 | 5×8 | **100.0 %** | 60.0 % |

### 1. The threshold was 0.0 — fixed

`config_infer.txt` never set `segmentation-threshold`, so nvinfer's
zero-initialised default of `0.0` was copied into
`NvOSD_MaskParams.threshold` (`gstnvinfer_meta_utils.cpp:213`). The parser
writes **sigmoid** values, which are all strictly greater than 0, so every
single mask cell passed and nvdsosd filled the whole box. That is the coloured
rectangle — it *was* the mask, at 100 % coverage.

One line, already applied:

```ini
segmentation-threshold=0.5
```

No rebuild needed. Re-run and you get silhouettes.

### 2. Masks are quarter-resolution — optional patch

The ONNX graph is `images [1,3,640,640] -> output1 [1,32,160,160]`. The
prototype grid is 160×160 against a 640×640 input, so one mask cell covers
4×4 network pixels — and against the 1280×720 display frame, roughly 8×4.5
display pixels. A mustard bottle gets a **5×10** mask stretched over a 14×33
box. Thresholding alone gives you a real shape, but a blocky one.

Your `.pt` path does not have this problem because `Detector._detect_builtin`
passes `retina_masks=True`, which upsamples the prototype masks to the full
input resolution before contouring. `patch_fullres_mask.txt` is the
DeepStream equivalent: bilinear-upsample the bbox-local mask to the box's
pixel size inside the parser, so nvdsosd receives a mask at object resolution
instead of prototype resolution.

Cost: a resize per object per frame, on CPU, inside the parser. At ~20 objects
a frame it is small next to the 22 ms the network already takes, but measure
it before shipping — the parser runs in the inference thread.

### 3. RGBA before nvdsosd — not your bug

Every NVIDIA reference pipeline reads
`nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! nvdsosd`, and
`run_deepstream.py` goes `tracker -> osd` directly. Worth aligning for
portability, but it is **not** what was hiding your masks: the translucent
fills prove nvdsosd is already blending correctly on this platform.

## Getting the polygon in Python (the `masks.xy` equivalent)

`.pt` gives you `results[0].masks.xy` — contour points. The DeepStream
equivalent reads `mask_params` off the object metadata in the probe. The mask
is a float array sized to the object's bbox, so contours come back in
bbox-local coordinates and need offsetting by the box origin:

```python
import numpy as np, cv2

def polygon_for(obj, rect):
    mp = obj.mask_params                     # NvOSD_MaskParams
    if mp is None or mp.size == 0:
        return None
    m = np.frombuffer(mp.data, dtype=np.float32, count=mp.width * mp.height)
    m = (m.reshape(mp.height, mp.width) > mp.threshold).astype(np.uint8)
    # mask is bbox-local: scale to the box, then offset into frame coords
    m = cv2.resize(m, (int(rect.width), int(rect.height)),
                   interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    c[:, 0] += rect.left
    c[:, 1] += rect.top
    return c
```

Two cautions, both learned the hard way in this pipeline:

* Read `mask_params` inside the **same single pass** over
  `frame_meta.object_items` that reads `rect_params`. The pyservicemaker
  wrapper is only valid while its iterator is advancing; holding a reference
  past the loop (or calling `list()` on it) segfaults the process.
* Copy the bytes out (`np.frombuffer(...).copy()` if you keep them) — the
  buffer is owned by the metadata and freed with it.

This gives you the tight contour bbox that `Detector` already computes from
`masks.xy` via `cv2.boundingRect`, so the DeepStream path can produce the same
tight boxes the `.pt` path does.
