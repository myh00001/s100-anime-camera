# s100-anime-camera optimization notes (local, not pushed)

## Architecture (upstream 71b857b)
- USB V4L2 capture thread (latest frame, prefers max size then MJPG).
- 2 spawn workers, gated: next frame admitted only after previous BPU chain ends (`on_bpu_chain_done` after last face parse / no-face detect). Max 1 in-flight BPU chain, max 2 frames in flight (one on BPU, one on CPU tail).
- Per frame: resize ≤1280 → scene AnimeGANv3 Ghibli (NHWC 640×360) → YuNet detect ≤3 faces → for each: warpAffine 512 crop → structure (AnimeGAN2, NCHW) → anime portrait → head parse → anime proportions / skin / brows → blend → finish_render (LAB chroma mix + Canny edges).
- Display: GTK fullscreen, flip horizontal. Optional FastAPI on 127.0.0.1:7860 (browser cam, separate path).
- Models: nash-e HBM only in upstream; this fork rebuilds nash-e/m/p from public ONNX (AnimeGANv3 Ghibli, face_paint_512_v2, Segformer face-parsing, YuNet). Upstream SHA256 pack NOT available → rebuilt models are functional substitutes, not bit-identical to published pack.

## Measured capture / codec (S100, USB 1080P cam)
| mode | fps | read_ms | dec_ms | CPU% |
|---|---:|---:|---:|---:|
| MJPG 1080p OpenCV decode | 25.9 | 38.6 | (in read) | 33.7 |
| MJPG 1080p raw + cv2.imdecode | 29.8 | 19.8 | 13.7 | 41.1 |
| MJPG 1080p raw + hobot_vio Decoder(MJPEG) | 30.9 | 29.9 | 2.53 | 2.8 |
| MJPG 720p + HW | 25.0 | 38.9 | 1.13 | 1.3 |
| YUYV 640x360 | 20.0 | 50.1 | — | 9.3 |

HW decode path (NV12→YCrCb full-range→BGR + resize≤1280): cos vs CPU 0.99999, PSNR 53.2 dB, mad 0.21; 9.4 ms vs CPU 16.4 ms.

## Encode (S100, 960×544)
cv2 JPEG Q85: 4.8 ms / 79 KB. HW MJPEG: 0.73 ms / 39 KB. HW H264: 1.48 ms / 4.4 KB. (Encoder rejects odd heights; pad to 544.)

## Model-only hrt_model_exec (O0 int8 rebuilds, 100 frames)
| board | model | thr1 ms | thr1 FPS | thr2 FPS |
|---|---|---:|---:|---:|
| S100 nash-e | scene_640 | 25.76 | 38.8 | 39.3 |
| S100 nash-e | portrait_anime_512 | 34.80 | 28.7 | 29.0 |
| S100P nash-m | scene_640 | 25.43 | 39.3 | 39.8 |
| S600 nash-p | scene_640 | 10.06 | 99.3 | 132.5 |
| S600 nash-p | portrait_anime_512 | 13.98 | 71.5 | 92.9 |

## Code changes in this tree (service/)
1. `fast_capture.py` + `local_display.py`: MJPEG via hobot_vio Decoder (env `RAC_HWDEC=1`).
2. `inference_pipeline.py`: `RAC_WORKERS` / `RAC_INFLIGHT` / `RAC_PRESCALE` / `RAC_THREADS` (defaults = upstream behaviour).
3. `board_engine.py`: parser labels as uint8 so `label_in` uses LUT (exact same argmax).

## Pending
- Finish portrait_structure + head_parse HBM for e/m/p; e2e serial + pipe baseline & opt on 3 boards.
- NV12 preprocess inside HBM (input_type_rt nv12) — separate compile variant.
- Zero-copy hbmem buffers end-to-end.

## No-face path (scene_640 O0 rebuild + YuNet + finish_render), 1080p cam frames resized ≤1280 → out 960×540
Measured 22:04–22:17 Asia/Shanghai. Upstream gating = W=2 INF=1.

| board | serial FPS (ms) | pipe W2/INF1 FPS (lat ms) | pipe W2/INF2 FPS (lat) | pipe W3/INF2 FPS (lat) |
|---|---:|---:|---:|---:|
| S100 nash-e | 7.39 (135) | 8.15 (119) | **14.06 (136)** | 14.00 (138) |
| S100P nash-m | 9.08 (110) | 9.93 (98) | **16.48 (117)** | 16.49 (117) |
| S600 nash-p | 11.26 (89) | 12.65 (78) | **23.99 (81)** | 23.97 (82) |

Raising INF from 1→2 (allow next BPU while previous CPU tail runs) nearly doubles FPS; a 3rd worker adds nothing (BPU saturated). Latency rises modestly.

## Capture/encode opts (S100) — ready in `service/fast_capture.py`, env RAC_HWDEC=1
See tables above. Full-range NV12→BGR cos 0.99999 vs OpenCV MJPEG path.

## Still blocked for full face path
portrait_structure_512 + head_parse_512 HBMs still compiling (structure nashe O0 in hbdk ~35min @100% bar, 24GB RSS). Face e2e baseline deferred.

## 23:12 resume checkpoint
- **portrait_structure_512 HBM**: abandoned for now. hbdk.compile hangs ~20–47 min at 100% bar / ~24 GB RSS on nash-e and nash-p (with or without InstanceNorm int16). Killed repeatedly.
- **Hybrid face path**: `board_engine.py` loads `portrait_structure_512.onnx` via onnxruntime CPU when `.hbm` missing (`ORT_STRUCTURE` log). Scene / anime / parse stay on BPU.
- **head_parse_512**: nash-e O0 compiling from 23:10 (chain then nash-p, then nash-m parse+anime).
- Ready HBMs: scene×3 marches, portrait_anime nashe+nashp (nashm anime still missing from earlier GPU OOM).
- SD int16 (unetA/cnA/unetD) S600 e2e: CFG img cos **0.93–0.95** (was ~0.70–0.80 int8); ~5× slower than O2 int8. See SD_BPU_RESULTS.md.

## 23:31 checkpoint (Asia/Shanghai)
- structure HBM: confirmed abandoned (hit 100% bar, never wrote .hbm; same for nash-p). Hybrid ORT_STRUCTURE deployed.
- head_parse_512 nashe O0: calib+PTQ+export OK (logits cos ~0.9987); hbdk.compile reached 100% @23:26, still no .hbm @23:31 (RSS frozen ~14.9GB, main TID still runnable). Same hang pattern as structure.
- Prepared `retry_parse_j1.sh` (jobs=1) + ORT_PARSER fallback in board_engine (head_parse.onnx 340MB — S100 RAM tight, prefer HBM).
- face_e2e nohup driver waiting: `~/racam/drivers/face_e2e_when_ready.sh` (pid logged).
- ORT installed: S100 1.23.2, S100P 1.19.2, S600 1.19.2 (+venv 1.31).
- SD int16 CFG table already in SD_BPU_RESULTS.md (no change).

## 23:37–23:38 (Asia/Shanghai)
- **head_parse_512_nashe_O0 HBM READY** (hb_compile completed after ~27 min total; hung-looking 100% bar for ~11 min then finished).
- S600 face serial **hybrid ORT structure+parser** (BPU scene+anime only): **0.084 fps / 11939 ms** (bpu 48.6 ms: scene 14.6 + anime 20.6; cpu/ORT ~11876 ms). faces_mean 1.65. Grids: `/home/guosheng/anime_view/repo_face_s600_{grid,000–003}.jpg`.
- Deploying nashe parse HBM to S100; nash-p/m parse chain continuing.

## 00:04 watchdog (Asia/Shanghai)
- Not restarted: nashp hb_compile (~30m, ~16GB), drive_parse_rest, face_e2e_when_ready (waiting nashp i≈61).
- Added `~/racam/drivers/face_watchdog.sh` (nohup): requeues nashp/nashm only if dead without HBM; relaunches e2e if it exits early.
- face_e2e wait limit extended in script file for any relaunch (360×30s).

## face e2e auto 2026-10-23 00:37:06 CST
```
{"mode": "serial_face", "n": 20, "fps": 0.13, "ms_mean": 7712.9, "bpu_ms": 138.0, "cpu_ms": 7559.7, "faces_mean": 1.65, "bpu_per_model_ms": {"scene_640": 14.5, "portrait_anime_512": 20.4, "head_parse_512": 54.4}}
{"mode": "pipe_w2_inf2_face", "n": 4, "fps": 0.17, "lat_ms": 4764.0, "proc_ms": 4759.8, "faces_mean": 1.0}
{"mode": "serial_face", "n": 20, "fps": 0.119, "ms_mean": 8425.8, "bpu_ms": 260.7, "cpu_ms": 8149.3, "faces_mean": 1.65, "bpu_per_model_ms": {"scene_640": 29.2, "portrait_anime_512": 40.2, "head_parse_512": 100.1}}
{"mode": "pipe_w2_inf2_face", "n": 4, "fps": 0.194, "lat_ms": 5310.6, "proc_ms": 5303.9, "faces_mean": 1.0}
```


## Ship 2026-10-23 (Asia/Shanghai)
- Default service: `RAC_FACE=0` scene-only + `RAC_HWDEC=1` + W2/INF2 for ~14 fps usable demo.
- Face still via `RAC_FACE=1` (ORT structure if no HBM).
- See `RUN_S100.md`.
