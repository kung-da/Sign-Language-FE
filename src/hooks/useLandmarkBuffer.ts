import { useCallback, useRef } from "react";
import type { WorkerLandmarks } from "./useMediaPipeLandmarks";
import { extractFrameFeatures, preprocessSequence, type FrameFeatures } from "../utils/landmarkPreprocessing";

const SEQ_LEN = 60;

export function useLandmarkBuffer() {
  const bufferRef = useRef<FrameFeatures[]>([]);

  const addFrame = useCallback((landmarks: WorkerLandmarks): void => {
    const frame = extractFrameFeatures(landmarks);
    const buffer = bufferRef.current;
    if (buffer.length < SEQ_LEN) {
      buffer.push(frame);
    }
  }, []);

  const isReady = useCallback((): boolean => {
    return bufferRef.current.length >= SEQ_LEN;
  }, []);

  const getSequence = useCallback((): number[][] => {
    return preprocessSequence(bufferRef.current.slice(0, SEQ_LEN));
  }, []);

  const reset = useCallback((): void => {
    bufferRef.current = [];
  }, []);

  const getProgress = useCallback((): number => {
    return bufferRef.current.length;
  }, []);

  return { addFrame, isReady, getSequence, reset, getProgress };
}
