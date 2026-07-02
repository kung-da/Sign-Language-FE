import type { LandmarkLike, WorkerLandmarks } from "../hooks/useMediaPipeLandmarks";

const EPSILON = 1e-6;

const POSE_LANDMARK_COUNT = 33;
const HAND_LANDMARK_COUNT = 21;
const FACE_BLENDSHAPE_COUNT = 52;
const MOUTH_LANDMARK_INDICES = [0, 17, 61, 291, 39, 269, 13, 14, 78, 308, 81, 311];
const MOUTH_LEFT_CORNER_LOCAL = MOUTH_LANDMARK_INDICES.indexOf(61);
const MOUTH_RIGHT_CORNER_LOCAL = MOUTH_LANDMARK_INDICES.indexOf(291);
const MOUTH_LANDMARK_COUNT = MOUTH_LANDMARK_INDICES.length;

const POSE_VALUES_PER_LANDMARK = 5;
const HAND_VALUES_PER_LANDMARK = 3;
const MOUTH_VALUES_PER_LANDMARK = 3;
const POSE_NORM_DIM = POSE_LANDMARK_COUNT * POSE_VALUES_PER_LANDMARK;
const POSE_WORLD_DIM = POSE_LANDMARK_COUNT * POSE_VALUES_PER_LANDMARK;
const POSE_FEATURE_DIM = POSE_NORM_DIM + POSE_WORLD_DIM;
const HAND_BLOCK_DIM = HAND_LANDMARK_COUNT * HAND_VALUES_PER_LANDMARK;
const HANDS_FEATURE_DIM = HAND_BLOCK_DIM * 4;
const MOUTH_FEATURE_DIM = MOUTH_LANDMARK_COUNT * MOUTH_VALUES_PER_LANDMARK;
export const LANDMARK_FEATURE_DIM = POSE_FEATURE_DIM + HANDS_FEATURE_DIM + FACE_BLENDSHAPE_COUNT + MOUTH_FEATURE_DIM;

const LEFT_HAND_NORM_START = 0;
const RIGHT_HAND_NORM_START = HAND_BLOCK_DIM;
const LEFT_HAND_WORLD_START = HAND_BLOCK_DIM * 2;
const RIGHT_HAND_WORLD_START = HAND_BLOCK_DIM * 3;

const VALID_LEFT_HAND = 1;
const VALID_RIGHT_HAND = 2;

const HAND_SIDE_MINORITY_RATIO_THRESHOLD = 0.2;
const INTERPOLATE_EDGE_MISSING_HANDS = false;

export interface FrameFeatures {
  face: number[];
  hands: number[];
  mouth: number[];
  pose: number[];
  validMask: number[];
}

export function extractFrameFeatures(landmarks: WorkerLandmarks): FrameFeatures {
  const poseNorm = poseToVector(landmarks.pose[0]);
  const poseWorld = poseToVector(landmarks.poseWorldLandmarks[0]);
  const { hands, validLeftHand, validRightHand } = extractHandsFeatures(landmarks);
  const face = faceBlendshapesToVector(landmarks.faceBlendshapes);
  const mouth = mouthLandmarksToVector(landmarks.face[0]);
  const hasFace = landmarks.face.length > 0 || face.some((value) => Math.abs(value) > EPSILON);

  return {
    face,
    hands,
    mouth,
    pose: [...poseNorm, ...poseWorld],
    validMask: [landmarks.pose.length > 0 ? 1 : 0, validLeftHand, validRightHand, hasFace ? 1 : 0],
  };
}

export function preprocessSequence(frames: FrameFeatures[]) {
  return preprocessFrameFeatures(frames).sequence;
}

export function preprocessFrameFeatures(frames: FrameFeatures[]) {
  const pose = frames.map((frame) => frame.pose.slice());
  const hands = frames.map((frame) => frame.hands.slice());
  const face = frames.map((frame) => frame.face.slice());
  const mouth = frames.map((frame) => frame.mouth.slice());
  const validMask = frames.map((frame) => frame.validMask.slice());
  const { hands: stableHands, validMask: stableValidMask } = stabilizeSingleHandSides(hands, validMask);
  const { processedHands, previewHands } = preprocessHandsSequence(stableHands, stableValidMask);
  const processedPose = normalizePoseSequence(pose);
  const processedMouth = normalizeMouthSequence(mouth);

  return {
    previewHands,
    sequence: frames.map((_, index) => {
      const row = [...processedPose[index], ...processedHands[index], ...face[index], ...processedMouth[index]];
      if (row.length !== LANDMARK_FEATURE_DIM) {
        return row.length > LANDMARK_FEATURE_DIM ? row.slice(0, LANDMARK_FEATURE_DIM) : [...row, ...createZeroArray(LANDMARK_FEATURE_DIM - row.length)];
      }
      return row;
    }),
    validMask: stableValidMask,
  };
}

export function handPreviewToLandmarks(previewHands: number[], validMask: number[]) {
  const hands: LandmarkLike[][] = [];

  if (validMask[VALID_LEFT_HAND] === 1 || hasNonZeroValues(previewHands.slice(LEFT_HAND_NORM_START, LEFT_HAND_NORM_START + HAND_BLOCK_DIM))) {
    hands.push(handBlockToLandmarks(previewHands.slice(LEFT_HAND_NORM_START, LEFT_HAND_NORM_START + HAND_BLOCK_DIM)));
  }

  if (validMask[VALID_RIGHT_HAND] === 1 || hasNonZeroValues(previewHands.slice(RIGHT_HAND_NORM_START, RIGHT_HAND_NORM_START + HAND_BLOCK_DIM))) {
    hands.push(handBlockToLandmarks(previewHands.slice(RIGHT_HAND_NORM_START, RIGHT_HAND_NORM_START + HAND_BLOCK_DIM)));
  }

  return hands.filter((hand) => hand.length > 0);
}

function poseToVector(landmarks?: LandmarkLike[]) {
  const vector = createZeroArray(POSE_NORM_DIM);
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

  for (let index = 0; index < landmarks.hands.length; index += 1) {
    const handNorm = handToVector(landmarks.hands[index]);
    const handWorld = handToVector(landmarks.handWorldLandmarks[index]);
    const label = landmarks.handedness[index];

    if (label === "Left" && validLeftHand === 0) {
      copyBlock(handNorm, hands, LEFT_HAND_NORM_START);
      copyBlock(handWorld, hands, LEFT_HAND_WORLD_START);
      validLeftHand = 1;
    } else if (label === "Right" && validRightHand === 0) {
      copyBlock(handNorm, hands, RIGHT_HAND_NORM_START);
      copyBlock(handWorld, hands, RIGHT_HAND_WORLD_START);
      validRightHand = 1;
    } else if (validLeftHand === 0) {
      copyBlock(handNorm, hands, LEFT_HAND_NORM_START);
      copyBlock(handWorld, hands, LEFT_HAND_WORLD_START);
      validLeftHand = 1;
    } else if (validRightHand === 0) {
      copyBlock(handNorm, hands, RIGHT_HAND_NORM_START);
      copyBlock(handWorld, hands, RIGHT_HAND_WORLD_START);
      validRightHand = 1;
    }
  }

  return { hands, validLeftHand, validRightHand };
}

function faceBlendshapesToVector(blendshapes: number[]) {
  const vector = createZeroArray(FACE_BLENDSHAPE_COUNT);
  for (let index = 0; index < Math.min(blendshapes.length, FACE_BLENDSHAPE_COUNT); index += 1) {
    vector[index] = blendshapes[index] ?? 0;
  }
  return vector;
}

function mouthLandmarksToVector(landmarks?: LandmarkLike[]) {
  const vector = createZeroArray(MOUTH_FEATURE_DIM);
  if (!landmarks) return vector;

  for (let outputIndex = 0; outputIndex < MOUTH_LANDMARK_COUNT; outputIndex += 1) {
    const landmark = landmarks[MOUTH_LANDMARK_INDICES[outputIndex]];
    if (!landmark) continue;

    const base = outputIndex * MOUTH_VALUES_PER_LANDMARK;
    vector[base] = landmark.x;
    vector[base + 1] = landmark.y;
    vector[base + 2] = landmark.z;
  }

  return vector;
}

function normalizePoseSequence(pose: number[][]) {
  return pose.map((frame) => [
    ...normalizePoseBlock(frame.slice(0, POSE_NORM_DIM)),
    ...normalizePoseBlock(frame.slice(POSE_NORM_DIM, POSE_FEATURE_DIM)),
  ]);
}

function normalizePoseBlock(block: number[]) {
  const output = block.slice();
  const coords = landmarkBlockToPoints(output, POSE_LANDMARK_COUNT, POSE_VALUES_PER_LANDMARK);
  const hasPose = coords.some((point) => point.some((value) => Math.abs(value) > EPSILON));
  const neck = midpoint(coords[11], coords[12]);
  const head = midpoint(coords[7], coords[8]);
  const scale = distance(head, neck);

  if (!hasPose || scale <= EPSILON) return output;

  for (let index = 0; index < POSE_LANDMARK_COUNT; index += 1) {
    const base = index * POSE_VALUES_PER_LANDMARK;
    output[base] = (output[base] - neck[0]) / scale;
    output[base + 1] = (output[base + 1] - neck[1]) / scale;
    output[base + 2] = (output[base + 2] - neck[2]) / scale;
  }

  return output;
}

function preprocessHandsSequence(hands: number[][], validMask: number[][]) {
  const handsOut = hands.map((frame) => frame.slice());
  const previewHands = hands.map((frame) => frame.slice());
  const handBlocks = [
    { start: LEFT_HAND_NORM_START, validColumn: VALID_LEFT_HAND },
    { start: RIGHT_HAND_NORM_START, validColumn: VALID_RIGHT_HAND },
    { start: LEFT_HAND_WORLD_START, validColumn: VALID_LEFT_HAND },
    { start: RIGHT_HAND_WORLD_START, validColumn: VALID_RIGHT_HAND },
  ];

  for (const block of handBlocks) {
    const interpolated = interpolateHandBlock(
      handsOut.map((frame) => frame.slice(block.start, block.start + HAND_BLOCK_DIM)),
      validMask.map((frame) => frame[block.validColumn] === 1),
    );

    for (let frameIndex = 0; frameIndex < handsOut.length; frameIndex += 1) {
      const normalized = normalizeHandBlock(interpolated[frameIndex]);
      for (let dim = 0; dim < HAND_BLOCK_DIM; dim += 1) {
        previewHands[frameIndex][block.start + dim] = interpolated[frameIndex][dim];
        handsOut[frameIndex][block.start + dim] = normalized[dim];
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

function normalizeHandBlock(block: number[]) {
  if (!hasNonZeroValues(block)) return block.slice();
  const output = block.slice();
  const wrist = [output[0], output[1], output[2]];

  for (let index = 0; index < HAND_LANDMARK_COUNT; index += 1) {
    const base = index * HAND_VALUES_PER_LANDMARK;
    output[base] -= wrist[0];
    output[base + 1] -= wrist[1];
    output[base + 2] -= wrist[2];
  }

  return output;
}

function normalizeMouthSequence(mouth: number[][]) {
  return mouth.map((frame) => normalizeMouthBlock(frame));
}

function normalizeMouthBlock(block: number[]) {
  if (!hasNonZeroValues(block)) return block.slice();

  const output = block.slice();
  const points = landmarkBlockToPoints(output, MOUTH_LANDMARK_COUNT, MOUTH_VALUES_PER_LANDMARK);
  const leftCorner = points[MOUTH_LEFT_CORNER_LOCAL];
  const rightCorner = points[MOUTH_RIGHT_CORNER_LOCAL];
  const center = midpoint(leftCorner, rightCorner);
  const scale = distance(leftCorner, rightCorner);

  if (scale <= EPSILON) return output;

  for (let index = 0; index < MOUTH_LANDMARK_COUNT; index += 1) {
    const base = index * MOUTH_VALUES_PER_LANDMARK;
    output[base] = (output[base] - center[0]) / scale;
    output[base + 1] = (output[base + 1] - center[1]) / scale;
    output[base + 2] = (output[base + 2] - center[2]) / scale;
  }

  return output;
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
      copyRange(handsOut[frameIndex], minoritySlices.normStart, handsOut[frameIndex], dominantSlices.normStart, HAND_BLOCK_DIM);
      copyRange(handsOut[frameIndex], minoritySlices.worldStart, handsOut[frameIndex], dominantSlices.worldStart, HAND_BLOCK_DIM);
      validOut[frameIndex][dominantSlices.validColumn] = 1;
    }

    if (hasMinority) {
      fillRange(handsOut[frameIndex], minoritySlices.normStart, HAND_BLOCK_DIM, 0);
      fillRange(handsOut[frameIndex], minoritySlices.worldStart, HAND_BLOCK_DIM, 0);
      validOut[frameIndex][minoritySlices.validColumn] = 0;
    }
  }

  return { hands: handsOut, validMask: validOut };
}

function getHandSlices(side: "left" | "right") {
  if (side === "left") {
    return {
      normStart: LEFT_HAND_NORM_START,
      worldStart: LEFT_HAND_WORLD_START,
      validColumn: VALID_LEFT_HAND,
    };
  }

  return {
    normStart: RIGHT_HAND_NORM_START,
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

function landmarkBlockToPoints(block: number[], count: number, valuesPerLandmark: number) {
  return Array.from({ length: count }, (_, index) => {
    const base = index * valuesPerLandmark;
    return [block[base] ?? 0, block[base + 1] ?? 0, block[base + 2] ?? 0];
  });
}

function midpoint(a: number[], b: number[]) {
  return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
}

function distance(a: number[], b: number[]) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function hasNonZeroValues(values: number[]) {
  return values.some((value) => Math.abs(value) > EPSILON);
}

function createZeroArray(length: number) {
  return Array.from({ length }, () => 0);
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
