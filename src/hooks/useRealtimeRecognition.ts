import { useCallback, useEffect, useRef, useState } from "react";
import { recognitionService } from "../services/recognitionService";
import type { PredictionResult } from "../types/recognition";
import { useLandmarkBuffer } from "./useLandmarkBuffer";
import type { WorkerLandmarks } from "./useMediaPipeLandmarks";

const SEQ_LEN = 60;

export function useRealtimeRecognition(isRunning: boolean) {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [bufferProgress, setBufferProgress] = useState(0);
  const { addFrame, isReady, getSequence, reset, getProgress } = useLandmarkBuffer();
  const inferringRef = useRef(false);

  // Reset when recognition stops
  useEffect(() => {
    if (!isRunning) {
      reset();
      setBufferProgress(0);
      setPrediction(null);
      setIsLoading(false);
      inferringRef.current = false;
    }
  }, [isRunning, reset]);

  const onLandmarks = useCallback(
    (landmarks: WorkerLandmarks) => {
      if (!isRunning || inferringRef.current) return;

      addFrame(landmarks);
      setBufferProgress(getProgress());

      if (isReady()) {
        const sequence = getSequence();
        reset();
        setBufferProgress(0);

        inferringRef.current = true;
        setIsLoading(true);

        recognitionService
          .predictFromSequence(sequence, 5)
          .then((result) => {
            setPrediction(result);
          })
          .catch((error) => {
            console.error("Prediction failed:", error);
            setPrediction({
              label: "Error",
              text: "Backend unavailable",
              confidence: 0,
              status: "error",
              topPredictions: [],
              stats: { fps: 0, latencyMs: 0, modelStatus: "error" },
              updatedAt: new Date().toISOString(),
            });
          })
          .finally(() => {
            setIsLoading(false);
            inferringRef.current = false;
          });
      }
    },
    [isRunning, addFrame, getProgress, isReady, getSequence, reset],
  );

  return {
    prediction,
    isLoading,
    bufferProgress,
    bufferTotal: SEQ_LEN,
    onLandmarks,
  };
}
