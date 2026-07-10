import { useCallback, useRef } from "react";
import type { WorkerLandmarks } from "./useMediaPipeLandmarks";
import { extractFrameFeatures, preprocessSequence, type FrameFeatures } from "../utils/landmarkPreprocessing";

const WINDOW_FRAMES = 90;
const MIN_FRAMES = 30;

export function useLandmarkBuffer() {
  const bufferRef = useRef<FrameFeatures[]>([]);

  const addFrame = useCallback((landmarks: WorkerLandmarks): void => {
    const frame = extractFrameFeatures(landmarks);
    const buffer = bufferRef.current;
    buffer.push(frame);
    while (buffer.length > WINDOW_FRAMES) buffer.shift();
  }, []);

  const isReady = useCallback((): boolean => {
    return bufferRef.current.length >= MIN_FRAMES;
  }, []);

  const getSequence = useCallback((): number[][] => {
    return preprocessSequence(bufferRef.current.slice(-WINDOW_FRAMES));
  }, []);

  const reset = useCallback((): void => {
    bufferRef.current = [];
  }, []);

  const getProgress = useCallback((): number => {
    return bufferRef.current.length;
  }, []);

  return { addFrame, isReady, getSequence, reset, getProgress };
}
