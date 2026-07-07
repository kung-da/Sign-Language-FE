import type { LandmarkLike, WorkerLandmarks } from "../hooks/useMediaPipeLandmarks";

const EPSILON = 1e-6;

const POSE_LANDMARK_COUNT = 33;
const HAND_LANDMARK_COUNT = 21;
const POSE_VALUES_PER_LANDMARK = 5;
const HAND_VALUES_PER_LANDMARK = 3;
const POSE_FEATURE_DIM = POSE_LANDMARK_COUNT * POSE_VALUES_PER_LANDMARK;
const HAND_BLOCK_DIM = HAND_LANDMARK_COUNT * HAND_VALUES_PER_LANDMARK;
const HANDS_FEATURE_DIM = HAND_BLOCK_DIM * 2;

export const LANDMARK_FEATURE_DIM = POSE_FEATURE_DIM + HANDS_FEATURE_DIM;

const LEFT_HAND_WORLD_START = 0;
const RIGHT_HAND_WORLD_START = HAND_BLOCK_DIM;

const VALID_LEFT_HAND = 1;
const VALID_RIGHT_HAND = 2;

const HAND_SIDE_MINORITY_RATIO_THRESHOLD = 0.2;
const INTERPOLATE_EDGE_MISSING_HANDS = false;

export interface FrameFeatures {
  hands: number[];
  pose: number[];
  validMask: number[];
}

export function extractFrameFeatures(landmarks: WorkerLandmarks): FrameFeatures {
  const poseWorld = poseToVector(landmarks.poseWorldLandmarks[0]);
  const { hands, validLeftHand, validRightHand } = extractHandsFeatures(landmarks);

  return {
    hands,
    pose: poseWorld,
    validMask: [landmarks.poseWorldLandmarks.length > 0 ? 1 : 0, validLeftHand, validRightHand],
  };
}

export function preprocessSequence(frames: FrameFeatures[]) {
  return preprocessFrameFeatures(frames).sequence;
}

export function preprocessFrameFeatures(frames: FrameFeatures[]) {
  const pose = frames.map((frame) => fitVector(frame.pose.slice(), POSE_FEATURE_DIM));
  const hands = frames.map((frame) => fitVector(frame.hands.slice(), HANDS_FEATURE_DIM));
  const validMask = frames.map((frame) => frame.validMask.slice());
  const { hands: stableHands, validMask: stableValidMask } = stabilizeSingleHandSides(hands, validMask);
  const { processedHands, previewHands } = preprocessHandsSequence(stableHands, stableValidMask);

  return {
    previewHands,
    sequence: frames.map((_, index) => fitVector([...pose[index], ...processedHands[index]], LANDMARK_FEATURE_DIM)),
    validMask: stableValidMask,
  };
}

export function handPreviewToLandmarks(previewHands: number[], validMask: number[]) {
  const hands: LandmarkLike[][] = [];

  const leftHand = previewHands.slice(LEFT_HAND_WORLD_START, LEFT_HAND_WORLD_START + HAND_BLOCK_DIM);
  if (validMask[VALID_LEFT_HAND] === 1 || hasNonZeroValues(leftHand)) {
    hands.push(handBlockToLandmarks(leftHand));
  }

  const rightHand = previewHands.slice(RIGHT_HAND_WORLD_START, RIGHT_HAND_WORLD_START + HAND_BLOCK_DIM);
  if (validMask[VALID_RIGHT_HAND] === 1 || hasNonZeroValues(rightHand)) {
    hands.push(handBlockToLandmarks(rightHand));
  }

  return hands.filter((hand) => hand.length > 0);
}

function poseToVector(landmarks?: LandmarkLike[]) {
  const vector = createZeroArray(POSE_FEATURE_DIM);
  if (!landmarks) return vector;

  for (let index = 0; index < Math.min(landmarks.length, POSE_LANDMARK_COUNT); index += 1) {
    const landmark = landmarks[index];
    const base = index * POSE_VALUES_PER_LANDMARK;
    vector[base] = landmark.x;
    vector[base + 1] = landmark.y;
    vector[base + 2] = landmark.z;
    vector[base + 3] = landmark.visibility ?? 0;
    vector[base + 4] = landmark.presence ?? 0;
  }

  return vector;
}

function handToVector(landmarks?: LandmarkLike[]) {
  const vector = createZeroArray(HAND_BLOCK_DIM);
  if (!landmarks) return vector;

  for (let index = 0; index < Math.min(landmarks.length, HAND_LANDMARK_COUNT); index += 1) {
    const landmark = landmarks[index];
    const base = index * HAND_VALUES_PER_LANDMARK;
    vector[base] = landmark.x;
    vector[base + 1] = landmark.y;
    vector[base + 2] = landmark.z;
  }

  return vector;
}

function extractHandsFeatures(landmarks: WorkerLandmarks) {
  const hands = createZeroArray(HANDS_FEATURE_DIM);
  let validLeftHand = 0;
  let validRightHand = 0;

  for (let index = 0; index < landmarks.handWorldLandmarks.length; index += 1) {
    const handWorld = handToVector(landmarks.handWorldLandmarks[index]);
    const label = landmarks.handedness[index];

    if (label === "Left" && validLeftHand === 0) {
      copyBlock(handWorld, hands, LEFT_HAND_WORLD_START);
      validLeftHand = 1;
    } else if (label === "Right" && validRightHand === 0) {
      copyBlock(handWorld, hands, RIGHT_HAND_WORLD_START);
      validRightHand = 1;
    } else if (validLeftHand === 0) {
      copyBlock(handWorld, hands, LEFT_HAND_WORLD_START);
      validLeftHand = 1;
    } else if (validRightHand === 0) {
      copyBlock(handWorld, hands, RIGHT_HAND_WORLD_START);
      validRightHand = 1;
    }
  }

  return { hands, validLeftHand, validRightHand };
}

function preprocessHandsSequence(hands: number[][], validMask: number[][]) {
  const handsOut = hands.map((frame) => frame.slice());
  const previewHands = hands.map((frame) => frame.slice());
  const handBlocks = [
    { start: LEFT_HAND_WORLD_START, validColumn: VALID_LEFT_HAND },
    { start: RIGHT_HAND_WORLD_START, validColumn: VALID_RIGHT_HAND },
  ];

  for (const block of handBlocks) {
    const interpolated = interpolateHandBlock(
      handsOut.map((frame) => frame.slice(block.start, block.start + HAND_BLOCK_DIM)),
      validMask.map((frame) => frame[block.validColumn] === 1),
    );

    for (let frameIndex = 0; frameIndex < handsOut.length; frameIndex += 1) {
      for (let dim = 0; dim < HAND_BLOCK_DIM; dim += 1) {
        const value = interpolated[frameIndex][dim] ?? 0;
        previewHands[frameIndex][block.start + dim] = value;
        handsOut[frameIndex][block.start + dim] = value;
      }
    }
  }

  return { previewHands, processedHands: handsOut };
}

function interpolateHandBlock(handBlock: number[][], validHint: boolean[]) {
  const output = handBlock.map((frame) => frame.slice());
  const validIndices = output
    .map((frame, index) => ({ frame, index }))
    .filter(({ frame, index }) => validHint[index] && hasNonZeroValues(frame))
    .map(({ index }) => index);

  if (validIndices.length === 0) return output;

  if (!INTERPOLATE_EDGE_MISSING_HANDS) {
    if (validIndices.length === 1) return output;
    const firstValid = validIndices[0];
    const lastValid = validIndices[validIndices.length - 1];

    for (let frameIndex = firstValid; frameIndex <= lastValid; frameIndex += 1) {
      output[frameIndex] = interpolateFrameAt(frameIndex, output, validIndices);
    }
    return output;
  }

  if (validIndices.length === 1) {
    return output.map(() => output[validIndices[0]].slice());
  }

  for (let frameIndex = 0; frameIndex < output.length; frameIndex += 1) {
    output[frameIndex] = interpolateFrameAt(frameIndex, output, validIndices);
  }
  return output;
}

function interpolateFrameAt(frameIndex: number, handBlock: number[][], validIndices: number[]) {
  const firstValid = validIndices[0];
  const lastValid = validIndices[validIndices.length - 1];
  if (frameIndex <= firstValid) return handBlock[firstValid].slice();
  if (frameIndex >= lastValid) return handBlock[lastValid].slice();
  if (validIndices.includes(frameIndex)) return handBlock[frameIndex].slice();

  let nextPosition = 1;
  while (validIndices[nextPosition] < frameIndex) nextPosition += 1;
  const previousIndex = validIndices[nextPosition - 1];
  const nextIndex = validIndices[nextPosition];
  const ratio = (frameIndex - previousIndex) / (nextIndex - previousIndex);

  return handBlock[previousIndex].map((previousValue, dim) => {
    const nextValue = handBlock[nextIndex][dim];
    return previousValue + (nextValue - previousValue) * ratio;
  });
}

function stabilizeSingleHandSides(hands: number[][], validMask: number[][]) {
  const handsOut = hands.map((frame) => frame.slice());
  const validOut = validMask.map((frame) => frame.slice());
  const leftCount = validOut.filter((frame) => frame[VALID_LEFT_HAND] === 1).length;
  const rightCount = validOut.filter((frame) => frame[VALID_RIGHT_HAND] === 1).length;

  if (leftCount === 0 || rightCount === 0) return { hands: handsOut, validMask: validOut };

  const dominant = leftCount >= rightCount ? "left" : "right";
  const minority = dominant === "left" ? "right" : "left";
  const dominantCount = Math.max(leftCount, rightCount);
  const minorityCount = Math.min(leftCount, rightCount);
  const minorityRatio = minorityCount / Math.max(dominantCount, 1);

  if (minorityRatio > HAND_SIDE_MINORITY_RATIO_THRESHOLD) return { hands: handsOut, validMask: validOut };

  const dominantSlices = getHandSlices(dominant);
  const minoritySlices = getHandSlices(minority);

  for (let frameIndex = 0; frameIndex < handsOut.length; frameIndex += 1) {
    const hasMinority = validOut[frameIndex][minoritySlices.validColumn] === 1;
    const hasDominant = validOut[frameIndex][dominantSlices.validColumn] === 1;

    if (hasMinority && !hasDominant) {
      copyRange(handsOut[frameIndex], minoritySlices.worldStart, handsOut[frameIndex], dominantSlices.worldStart, HAND_BLOCK_DIM);
      validOut[frameIndex][dominantSlices.validColumn] = 1;
    }

    if (hasMinority) {
      fillRange(handsOut[frameIndex], minoritySlices.worldStart, HAND_BLOCK_DIM, 0);
      validOut[frameIndex][minoritySlices.validColumn] = 0;
    }
  }

  return { hands: handsOut, validMask: validOut };
}

function getHandSlices(side: "left" | "right") {
  if (side === "left") {
    return {
      worldStart: LEFT_HAND_WORLD_START,
      validColumn: VALID_LEFT_HAND,
    };
  }

  return {
    worldStart: RIGHT_HAND_WORLD_START,
    validColumn: VALID_RIGHT_HAND,
  };
}

function handBlockToLandmarks(block: number[]) {
  if (!hasNonZeroValues(block)) return [];

  return Array.from({ length: HAND_LANDMARK_COUNT }, (_, index) => {
    const base = index * HAND_VALUES_PER_LANDMARK;
    return {
      x: block[base] ?? 0,
      y: block[base + 1] ?? 0,
      z: block[base + 2] ?? 0,
      visibility: 1,
    };
  });
}

function hasNonZeroValues(values: number[]) {
  return values.some((value) => Math.abs(value) > EPSILON);
}

function createZeroArray(length: number) {
  return Array.from({ length }, () => 0);
}

function fitVector(values: number[], length: number) {
  if (values.length === length) return values;
  return values.length > length ? values.slice(0, length) : [...values, ...createZeroArray(length - values.length)];
}

function copyBlock(source: number[], target: number[], targetStart: number) {
  for (let index = 0; index < source.length; index += 1) {
    target[targetStart + index] = source[index] ?? 0;
  }
}

function copyRange(source: number[], sourceStart: number, target: number[], targetStart: number, length: number) {
  const values = source.slice(sourceStart, sourceStart + length);
  for (let index = 0; index < values.length; index += 1) {
    target[targetStart + index] = values[index] ?? 0;
  }
}

function fillRange(values: number[], start: number, length: number, value: number) {
  for (let index = start; index < start + length; index += 1) {
    values[index] = value;
  }
}
