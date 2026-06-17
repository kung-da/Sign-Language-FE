# VSL AI Translator Frontend

Frontend demo cho hệ thống nhận diện Ngôn ngữ Ký hiệu Việt Nam theo thời gian thực. Ứng dụng dùng React, Vite, Tailwind CSS và MediaPipe Tasks Vision để hiển thị camera, vẽ landmark tay/khuôn mặt/tư thế trực tiếp trên trình duyệt, đồng thời mô phỏng kết quả nhận diện bằng dữ liệu mock.

## Tính Năng Chính

- Bật webcam và hiển thị preview 1280x720.
- Chạy MediaPipe hand, face và pose landmark trên browser.
- Tách inference MediaPipe sang 3 Web Worker riêng để giảm nghẽn main thread.
- Dùng model và wasm local trong `public/mediapipe`, hạn chế phụ thuộc mạng khi chạy camera.
- Overlay landmark trực tiếp lên canvas.
- Panel đo hiệu năng pipeline khi chạy webcam.
- Panel mock prediction, confidence, top predictions, FPS và latency.
- Modal thêm ký hiệu mới, hiện đang lưu mock payload.
- Các section trình bày pipeline, architecture, metrics, features và social impact.

## Công Nghệ

- React 18 + TypeScript
- Vite
- Tailwind CSS
- Framer Motion
- Lucide React
- `@mediapipe/tasks-vision`

## Cài Đặt

```bash
npm install
```

## Chạy Local

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

## MediaPipe Local Assets

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

## Kiến Trúc Camera Hiện Tại

- `src/hooks/useCamera.ts`: xin quyền webcam, bật/tắt stream.
- `src/hooks/useMediaPipeLandmarks.ts`: resize frame vào canvas nhỏ, gửi frame sang worker, nhận landmark và vẽ overlay.
- `src/workers/mediaPipeLandmarks.worker.ts`: mỗi worker chạy một task MediaPipe (`hand`, `face`, hoặc `pose`) bằng CPU delegate.
- `src/components/demo/CameraPreview.tsx`: hiển thị video, canvas overlay, trạng thái MediaPipe và lỗi camera.

## Trạng Thái AI/Backend

Hiện tại phần landmark là thật, chạy local trên browser. Tuy nhiên phần nhận diện câu/ký hiệu vẫn là mock:

- `src/services/recognitionService.ts`
- `src/services/vectorDbService.ts`
- `src/hooks/useMockRecognition.ts`
- `src/data/mockPredictions.ts`

Khi backend thật sẵn sàng, thay các service mock bằng API tới Python/FastAPI hoặc pipeline inference thật.

## Đánh Giá Hiệu Năng Cần Có

Web hiện có panel `Pipeline Performance` ngay dưới camera preview. Panel này đo trực tiếp phần pipeline frontend có thể đo được và đánh dấu rõ những chỉ số cần model/backend thật.

| Chỉ số | Trạng thái hiện tại | Cần bổ sung khi có model thật |
| --- | --- | --- |
| MediaPipe extraction time | Đã có trong web. Đo thời gian xử lý landmark trong 3 worker và lấy giá trị lớn nhất của hand/face/pose. | Có thể lưu log theo từng video test để tính trung bình, min, max, p95. |
| Model inference time | Có ô hiển thị trong web nhưng đang là `Not measured`, vì model nhận diện ký hiệu thật chưa được nối. | Đo thời gian model phân loại/sequence model chạy sau khi có keypoint. |
| End-to-end latency | Đã có trong web cho pipeline camera -> MediaPipe -> overlay. | Khi nối model thật, mở rộng đến lúc có prediction cuối cùng. |
| FPS pipeline | Đã có trong web. Đếm số frame landmark hoàn tất mỗi giây. | Khi nối model thật, đổi thành FPS của pipeline đầy đủ camera -> MediaPipe -> model -> smoothing -> output. |
| Model size | Đã có trong web cho tổng kích thước các MediaPipe `.task` local và breakdown từng file. | Thêm kích thước model nhận diện ký hiệu sau huấn luyện, ví dụ `.onnx`, `.tflite`, `.pt`, `.pth` hoặc `.keras`. |
| Smoothing/voting | Có phần readiness trong web với đề xuất majority vote window. | Khi có prediction thật, đo số lần label đổi trước/sau smoothing trên cùng video test. |
| RAM usage | Đã có JS heap RAM nếu browser hỗ trợ `performance.memory`. | Đo thêm process memory bằng DevTools hoặc Task Manager nếu cần báo cáo hệ thống. |
| CPU/GPU usage | Có ô hiển thị trong web nhưng browser không cung cấp số đo chính xác, nên đang ghi `Not available`. | Đo bằng Chrome DevTools Performance, Windows Task Manager, GPU profiler hoặc backend profiler. |

### Gợi Ý Cách Đo

- **MediaPipe extraction time**: hiện đã đo trong từng worker bằng `performance.now()`.
- **Model inference time**: đặt timer quanh hàm predict của model thật.
- **End-to-end latency**: hiện đo từ lúc gửi frame vào pipeline đến lúc nhận đủ landmark để vẽ overlay.
- **FPS**: hiện đếm số frame landmark hoàn tất trong mỗi giây, không chỉ FPS của video preview.
- **Model size**: hiện đọc `Content-Length` của các model local trong `public/mediapipe/models`.
- **RAM/CPU/GPU usage**: ghi nhận bằng Chrome DevTools, Windows Task Manager hoặc công cụ profiler tương ứng.
- **Smoothing/voting**: so sánh số lần label bị đổi trước và sau khi áp dụng thuật toán trên cùng một đoạn video test.

### Công Thức Đề Xuất

```text
MediaPipe extraction time = t_landmarks_done - t_frame_sent_to_mediapipe
Model inference time      = t_model_output - t_model_input
End-to-end latency        = t_final_output - t_frame_received
Pipeline FPS              = completed_predictions / elapsed_seconds
```

## Ghi Chú Khi Test Camera

- Nên chạy trên `http://127.0.0.1` hoặc HTTPS để browser cho phép webcam.
- Nếu thấy `MediaPipe unavailable`, hãy restart dev server sau khi đổi `vite.config.ts`.
- Nếu trình duyệt đã cache server cũ, đổi port hoặc hard refresh.
- Fake camera trong Chrome có thể trả `Hands: 0 | Face: 0 | Pose: 0`; điều này vẫn xác nhận MediaPipe đã load và chạy.
