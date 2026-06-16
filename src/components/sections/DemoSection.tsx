import { useState } from "react";
import { AddNewSignModal } from "../demo/AddNewSignModal";
import { CameraPreview } from "../demo/CameraPreview";
import { PredictionPanel } from "../demo/PredictionPanel";
import { SectionHeading } from "../ui/SectionHeading";
import { useCamera } from "../../hooks/useCamera";
import { useMockRecognition } from "../../hooks/useMockRecognition";

export function DemoSection() {
  const { stream, isCameraActive, startCamera, stopCamera, error } = useCamera();
  const { prediction, isLoading } = useMockRecognition(isCameraActive);
  const [modalOpen, setModalOpen] = useState(false);

  return (
    <section id="demo" className="section-container">
      <SectionHeading
        eyebrow="Realtime Demo"
        title="Simulated webcam recognition workspace"
        description="Turn on the camera to simulate prediction updates, confidence smoothing, and a low-confidence add-sign workflow."
      />
      <div className="grid gap-6 lg:grid-cols-[1.35fr_0.65fr]">
        <CameraPreview
          stream={stream}
          isActive={isCameraActive}
          onStart={startCamera}
          onStop={stopCamera}
          error={error}
        />
        <PredictionPanel prediction={prediction} isLoading={isLoading} onAddSign={() => setModalOpen(true)} />
      </div>
      <AddNewSignModal open={modalOpen} onClose={() => setModalOpen(false)} />
    </section>
  );
}
