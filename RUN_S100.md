# S100 fast anime camera (no-face default)

Deploy path on board: `/root/s100-anime-camera-opt`

## Defaults (max speed)
| env | default | meaning |
|---|---|---|
| `RAC_HWDEC` | `1` | hobot_vio MJPEG decode |
| `RAC_WORKERS` | `2` | spawn workers |
| `RAC_INFLIGHT` | `2` | admit next BPU while prior CPU tail runs |
| `RAC_FACE` | `0` | **scene-only** (skip YuNet + face enhance) |
| `RAC_PRESCALE` | `1` | resize ≤1280 before IPC |

Face path: `RAC_FACE=1` (needs `portrait_anime_512.hbm`, `head_parse_512.hbm`, and `portrait_structure_512.hbm` **or** `.onnx` ORT fallback). Structure HBM unavailable → ORT ~0.1 fps.

## Start on S100
```bash
ssh s100
cd /root/s100-anime-camera-opt/service
./start.sh
# or face mode:
# RAC_FACE=1 ./start.sh
```

USB cam: `/dev/video0` (1080P MJPG).

## Measured FPS (S100 nash-e, 2026-10-23 17:35 Asia/Shanghai)
| mode | fps | lat/proc |
|---|---:|---|
| pipe W2/INF2 `RAC_FACE=0` (frames, shipped code) | **13.74** | 143 / 132 ms |
| serial live `/dev/video0` `RAC_FACE=0` | 7.44 | 134.5 ms |
| prior pipe W2/INF2 (same path) | 14.06 | 136 ms |

Samples: `/home/algodev/anime_view/noface_*.jpg`, `noface_grid.jpg`, `livecam_*.jpg` (also on board `/root/s100-anime-camera-opt/bench_out/`).

### GTK display note
Board currently has **no X11** (`/tmp/.X11-unix` empty). `./start.sh` needs a local desktop (`DISPLAY=:0`). Pipeline itself is verified via `live_bench_noface.py` / `cam_smoke.py`.

## Models required for default (RAC_FACE=0)
- `service/models/scene_640.hbm`
- `service/models/model-config.json`

Optional for `RAC_FACE=1`: `portrait_anime_512.hbm`, `head_parse_512.hbm`, `portrait_structure_512.onnx` (ORT), `face_detection_yunet_2023mar.onnx`.
