import { useEffect, useState } from "react";
import { recognitionService } from "../services/recognitionService";
import type { PredictionResult } from "../types/recognition";

export function useMockRecognition(isRunning: boolean) {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let isMounted = true;
    let intervalId: number | undefined;

    const updatePrediction = async () => {
      if (!isRunning) {
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      const nextPrediction = await recognitionService.getPrediction();
      if (isMounted) {
        setPrediction(nextPrediction);
        setIsLoading(false);
      }
    };

    updatePrediction();
    if (isRunning) {
      intervalId = window.setInterval(updatePrediction, 2200 + Math.random() * 800);
    }

    return () => {
      isMounted = false;
      if (intervalId) window.clearInterval(intervalId);
    };
  }, [isRunning]);

  return { prediction, isLoading };
}
