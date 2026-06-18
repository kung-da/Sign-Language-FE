import { useCallback, useRef } from "react";
import type { WorkerLandmarks } from "./useMediaPipeLandmarks";

/**
 * Target feature dimension expected by the TCN model.
 * Pose 33×(x,y,z) + LeftHand 21×(x,y,z) + RightHand 21×(x,y,z) + Face landmarks
 * The exact layout must match the training pipeline.
 *
 * If the flattened landmarks are shorter, zeros are appended.
 * If they are longer, the vector is truncated.
 */
const TARGET_DIM = 634;
const SEQ_LEN = 60;

interface NormalizedLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

function flattenLandmarks(landmarks: WorkerLandmarks): number[] {
  const values: number[] = [];

  // Pose: 33 landmarks × (x, y, z)
  const pose = landmarks.pose[0] ?? [];
  for (let i = 0; i < 33; i++) {
    const lm = pose[i] as NormalizedLandmark | undefined;
    values.push(lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0);
  }

  // Left hand: 21 landmarks × (x, y, z)
  const leftHand = landmarks.hands[0] ?? [];
  for (let i = 0; i < 21; i++) {
    const lm = leftHand[i] as NormalizedLandmark | undefined;
    values.push(lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0);
  }

  // Right hand: 21 landmarks × (x, y, z)
  const rightHand = landmarks.hands[1] ?? [];
  for (let i = 0; i < 21; i++) {
    const lm = rightHand[i] as NormalizedLandmark | undefined;
    values.push(lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0);
  }

  // Face landmarks: fill remaining dimensions
  const face = landmarks.face[0] ?? [];
  for (const lm of face) {
    values.push((lm as NormalizedLandmark).x ?? 0, (lm as NormalizedLandmark).y ?? 0, (lm as NormalizedLandmark).z ?? 0);
    if (values.length >= TARGET_DIM) break;
  }

  // Pad or truncate to TARGET_DIM
  while (values.length < TARGET_DIM) {
    values.push(0);
  }

  return values.slice(0, TARGET_DIM);
}

export function useLandmarkBuffer() {
  const bufferRef = useRef<number[][]>([]);

  const addFrame = useCallback((landmarks: WorkerLandmarks): void => {
    const frame = flattenLandmarks(landmarks);
    const buffer = bufferRef.current;
    if (buffer.length < SEQ_LEN) {
      buffer.push(frame);
    }
  }, []);

  const isReady = useCallback((): boolean => {
    return bufferRef.current.length >= SEQ_LEN;
  }, []);

  const getSequence = useCallback((): number[][] => {
    return bufferRef.current.slice(0, SEQ_LEN);
  }, []);

  const reset = useCallback((): void => {
    bufferRef.current = [];
  }, []);

  const getProgress = useCallback((): number => {
    return bufferRef.current.length;
  }, []);

  return { addFrame, isReady, getSequence, reset, getProgress };
}
