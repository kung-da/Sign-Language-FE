import { useState } from "react";
import { AddNewSignModal } from "../demo/AddNewSignModal";
import { CameraPreview } from "../demo/CameraPreview";
import { PredictionPanel } from "../demo/PredictionPanel";
import { SectionHeading } from "../ui/SectionHeading";
import { useCamera } from "../../hooks/useCamera";
import { useRealtimeRecognition } from "../../hooks/useRealtimeRecognition";
import { recognitionService } from "../../services/recognitionService";
import type { PredictionResult } from "../../types/recognition";

export function DemoSection() {
  const { stream, settings, isCameraActive, startCamera, stopCamera, error } = useCamera();
  const [isVideoActive, setIsVideoActive] = useState(false);
  const { prediction, isLoading, bufferProgress, bufferTotal, onLandmarks } =
    useRealtimeRecognition(isVideoActive);
  const [modalOpen, setModalOpen] = useState(false);
  const [uploadPrediction, setUploadPrediction] = useState<PredictionResult | null>(null);
  const [isUploadLoading, setIsUploadLoading] = useState(false);

  const handleVideoUpload = async (file: File) => {
    setUploadPrediction(null);
    setIsUploadLoading(true);
    try {
      const result = await recognitionService.predictFromVideo(file, 5);
      setUploadPrediction(result);
    } catch (uploadError) {
      console.error("Backend demo upload failed:", uploadError);
      setUploadPrediction({
        label: "Error",
        text: "Upload prediction failed",
        confidence: 0,
        status: "error",
        topPredictions: [],
        stats: { fps: 0, latencyMs: 0, modelStatus: "error" },
        updatedAt: new Date().toISOString(),
      });
    } finally {
      setIsUploadLoading(false);
    }
  };

  const handleStartCamera = () => {
    setUploadPrediction(null);
    void startCamera();
  };

  return (
    <section id="demo" className="section-container">
      <SectionHeading
        eyebrow="Realtime Demo"
        title="Webcam recognition workspace"
        description="Turn on the camera to extract MediaPipe keypoints and run baseline TCN inference via the backend API."
      />
      <div className="grid gap-6 lg:grid-cols-[1.35fr_0.65fr]">
        <CameraPreview
          stream={stream}
          cameraSettings={settings}
          isActive={isCameraActive}
          onStart={handleStartCamera}
          onStop={stopCamera}
          error={error}
          onLandmarks={onLandmarks}
          onVideoUpload={handleVideoUpload}
          onVideoActiveChange={setIsVideoActive}
        />
        <PredictionPanel
          bufferProgress={bufferProgress}
          bufferTotal={bufferTotal}
          prediction={uploadPrediction ?? prediction}
          isLoading={isUploadLoading || isLoading}
          onAddSign={() => setModalOpen(true)}
        />
      </div>
      <AddNewSignModal open={modalOpen} onClose={() => setModalOpen(false)} />
    </section>
  );
}
