# VSL AI Translator Frontend

Frontend demo cho hệ thống nhận diện Ngôn ngữ Ký hiệu Việt Nam theo thời gian thực. Ứng dụng dùng React, Vite, Tailwind CSS và MediaPipe Tasks Vision để hiển thị camera, vẽ landmark tay/khuôn mặt/tư thế trực tiếp trên trình duyệt, đồng thời mô phỏng kết quả nhận diện bằng dữ liệu mock.

## Tính năng chính

- Bật webcam và hiển thị preview 1280x720.
- Chạy MediaPipe hand, face và pose landmark trên browser.
- Tách inference sang 3 Web Worker riêng để giảm nghẽn main thread.
- Dùng model và wasm local trong `public/mediapipe`, hạn chế phụ thuộc mạng khi chạy camera.
- Overlay landmark trực tiếp lên canvas.
- Panel mock prediction, confidence, top predictions, FPS và latency.
- Modal thêm ký hiệu mới, hiện đang lưu mock payload.
- Các section trình bày pipeline, architecture, metrics, features và social impact.

## Công nghệ

- React 18 + TypeScript
- Vite
- Tailwind CSS
- Framer Motion
- Lucide React
- `@mediapipe/tasks-vision`

## Cài đặt

```bash
npm install
```

## Chạy local

```bash
npm run dev -- --host 127.0.0.1
```

Mở URL Vite in ra, thường là:

```text
http://127.0.0.1:5173/
```

Nếu port `5173` đang bị server cũ chiếm, tắt server đó hoặc chạy port khác:

```bash
npm run dev -- --host 127.0.0.1 --port 5175 --strictPort
```

## Build

```bash
npm run build
```

Build output nằm trong `dist/`.

## MediaPipe local assets

Ứng dụng cần các file sau:

```text
public/mediapipe/models/
  face_landmarker.task
  hand_landmarker.task
  pose_landmarker_lite.task

public/mediapipe/wasm/
  vision_wasm_internal.js
  vision_wasm_internal.wasm
  vision_wasm_module_internal.js
  vision_wasm_module_internal.wasm
  vision_wasm_nosimd_internal.js
  vision_wasm_nosimd_internal.wasm
```

Trong development, Vite config serve wasm runtime qua `/mediapipe-dev-wasm` để tránh lỗi import file JavaScript trực tiếp từ `public`. Trong production build, các file trong `public/mediapipe` được copy sang `dist/mediapipe`.

## Kiến trúc camera hiện tại

- `src/hooks/useCamera.ts`: xin quyền webcam, bật/tắt stream.
- `src/hooks/useMediaPipeLandmarks.ts`: resize frame vào canvas nhỏ, gửi frame sang worker, nhận landmark và vẽ overlay.
- `src/workers/mediaPipeLandmarks.worker.ts`: mỗi worker chạy một task MediaPipe (`hand`, `face`, hoặc `pose`) bằng CPU delegate.
- `src/components/demo/CameraPreview.tsx`: hiển thị video, canvas overlay, trạng thái MediaPipe và lỗi camera.

## Trạng thái AI/backend

Hiện tại phần landmark là thật, chạy local trên browser. Tuy nhiên phần nhận diện câu/ký hiệu vẫn là mock:

- `src/services/recognitionService.ts`
- `src/services/vectorDbService.ts`
- `src/hooks/useMockRecognition.ts`
- `src/data/mockPredictions.ts`

Khi backend thật sẵn sàng, thay các service mock bằng API tới Python/FastAPI hoặc pipeline inference thật.

## Ghi chú khi test camera

- Nên chạy trên `http://127.0.0.1` hoặc HTTPS để browser cho phép webcam.
- Nếu thấy `MediaPipe unavailable`, hãy restart dev server sau khi đổi `vite.config.ts`.
- Nếu trình duyệt đã cache server cũ, đổi port hoặc hard refresh.
- Fake camera trong Chrome có thể trả `Hands: 0 | Face: 0 | Pose: 0`; điều này vẫn xác nhận MediaPipe đã load và chạy.
