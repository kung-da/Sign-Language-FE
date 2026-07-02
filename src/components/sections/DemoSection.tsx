import { useState } from "react";
import { AddNewSignModal } from "../demo/AddNewSignModal";
import { CameraPreview } from "../demo/CameraPreview";
import { PredictionPanel } from "../demo/PredictionPanel";
import { SectionHeading } from "../ui/SectionHeading";
import { useCamera } from "../../hooks/useCamera";
import { useRealtimeRecognition } from "../../hooks/useRealtimeRecognition";

export function DemoSection() {
  const { stream, settings, isCameraActive, startCamera, stopCamera, error } = useCamera();
  const { prediction, isLoading, bufferProgress, bufferTotal, onLandmarks } =
    useRealtimeRecognition(isCameraActive);
  const [modalOpen, setModalOpen] = useState(false);

  return (
    <section id="demo" className="section-container">
      <SectionHeading
        eyebrow="Realtime Demo"
        title="Webcam recognition workspace"
        description="Turn on the camera to extract MediaPipe keypoints, buffer 60 frames, and run baseline TCN inference via the backend API."
      />
      <div className="grid gap-6 lg:grid-cols-[1.35fr_0.65fr]">
        <CameraPreview
          stream={stream}
          cameraSettings={settings}
          isActive={isCameraActive}
          onStart={startCamera}
          onStop={stopCamera}
          error={error}
          onLandmarks={onLandmarks}
          bufferProgress={bufferProgress}
          bufferTotal={bufferTotal}
        />
        <PredictionPanel prediction={prediction} isLoading={isLoading} onAddSign={() => setModalOpen(true)} />
      </div>
      <AddNewSignModal open={modalOpen} onClose={() => setModalOpen(false)} />
    </section>
  );
}
