import type { LandmarkLike, WorkerLandmarks } from "../hooks/useMediaPipeLandmarks";

const EPSILON = 1e-6;

const SEQ_LEN = 60;
const POSE_LANDMARK_COUNT = 33;
const HAND_LANDMARK_COUNT = 21;
const POSE_VALUES_PER_LANDMARK = 5;
const HAND_VALUES_PER_LANDMARK = 3;

const POSE_NORM_DIM = POSE_LANDMARK_COUNT * POSE_VALUES_PER_LANDMARK;
const POSE_WORLD_DIM = POSE_NORM_DIM;
const POSE_FEATURE_DIM = POSE_NORM_DIM + POSE_WORLD_DIM;

const LEFT_HAND_NORM_DIM = HAND_LANDMARK_COUNT * HAND_VALUES_PER_LANDMARK;
const RIGHT_HAND_NORM_DIM = LEFT_HAND_NORM_DIM;
const LEFT_HAND_WORLD_DIM = LEFT_HAND_NORM_DIM;
const RIGHT_HAND_WORLD_DIM = LEFT_HAND_NORM_DIM;
const HANDS_FEATURE_DIM = LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM + LEFT_HAND_WORLD_DIM + RIGHT_HAND_WORLD_DIM;

export const LANDMARK_FEATURE_DIM = POSE_WORLD_DIM + LEFT_HAND_WORLD_DIM + RIGHT_HAND_WORLD_DIM;

const LEFT_HAND_NORM_START = 0;
const RIGHT_HAND_NORM_START = LEFT_HAND_NORM_DIM;
const LEFT_HAND_WORLD_START = LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM;
const RIGHT_HAND_WORLD_START = LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM + LEFT_HAND_WORLD_DIM;

const VALID_POSE = 0;
const VALID_LEFT_HAND = 1;
const VALID_RIGHT_HAND = 2;
const VALID_FACE = 3;

const ENABLE_ACTION_TRIM = true;
const ACTION_TRIM_MARGIN_FRAMES = 3;
const ACTION_TRIM_MIN_KEEP_FRAMES = 12;
const ACTION_TRIM_HAND_MOTION_THRESHOLD = 0.008;
const ACTION_TRIM_POSE_MOTION_THRESHOLD = 0.006;
const HAND_SIDE_MINORITY_RATIO_THRESHOLD = 0.2;
const INTERPOLATE_EDGE_MISSING_HANDS = false;

export interface FrameFeatures {
  hands: number[];
  pose: number[];
  validMask: number[];
}

export function extractFrameFeatures(landmarks: WorkerLandmarks): FrameFeatures {
  const poseNorm = poseToVector(landmarks.pose[0]);
  const poseWorld = poseToVector(landmarks.poseWorldLandmarks[0]);
  const { hands, validLeftHand, validRightHand } = extractHandsFeatures(landmarks);

  return {
    hands,
    pose: [...poseNorm, ...poseWorld],
    validMask: [landmarks.pose.length > 0 ? 1 : 0, validLeftHand, validRightHand, landmarks.face.length > 0 ? 1 : 0],
  };
}

export function preprocessSequence(frames: FrameFeatures[]) {
  return preprocessFrameFeatures(frames).sequence;
}

export function preprocessFrameFeatures(frames: FrameFeatures[]) {
  const fittedFrames = frames.map((frame) => ({
    pose: fitVector(frame.pose.slice(), POSE_FEATURE_DIM),
    hands: fitVector(frame.hands.slice(), HANDS_FEATURE_DIM),
    validMask: fitVector(frame.validMask.slice(), VALID_FACE + 1),
  }));
  const trimmedFrames = trimActionFrames(fittedFrames);
  const sampledFrames = resampleFrames(trimmedFrames, SEQ_LEN);

  const pose = sampledFrames.map((frame) => frame.pose.slice());
  const hands = sampledFrames.map((frame) => frame.hands.slice());
  const validMask = sampledFrames.map((frame) => frame.validMask.slice());
  const { hands: stableHands, validMask: stableValidMask } = stabilizeSingleHandSides(hands, validMask);
  const normalizedPose = normalizePoseSequence(pose);
  const { processedHands, previewHands } = preprocessHandsSequence(stableHands, stableValidMask);

  return {
    previewHands,
    sequence: sampledFrames.map((_, index) =>
      fitVector(
        [
          ...normalizedPose[index].slice(POSE_NORM_DIM, POSE_FEATURE_DIM),
          ...processedHands[index].slice(LEFT_HAND_WORLD_START, LEFT_HAND_WORLD_START + LEFT_HAND_WORLD_DIM),
          ...processedHands[index].slice(RIGHT_HAND_WORLD_START, RIGHT_HAND_WORLD_START + RIGHT_HAND_WORLD_DIM),
        ],
        LANDMARK_FEATURE_DIM,
      ),
    ),
    validMask: stableValidMask,
  };
}

export function handPreviewToLandmarks(previewHands: number[], validMask: number[]) {
  const hands: LandmarkLike[][] = [];

  const leftHand = previewHands.slice(LEFT_HAND_NORM_START, LEFT_HAND_NORM_START + LEFT_HAND_NORM_DIM);
  if (validMask[VALID_LEFT_HAND] === 1 || hasNonZeroValues(leftHand)) {
    hands.push(handBlockToLandmarks(leftHand));
  }

  const rightHand = previewHands.slice(RIGHT_HAND_NORM_START, RIGHT_HAND_NORM_START + RIGHT_HAND_NORM_DIM);
  if (validMask[VALID_RIGHT_HAND] === 1 || hasNonZeroValues(rightHand)) {
    hands.push(handBlockToLandmarks(rightHand));
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
  const vector = createZeroArray(LEFT_HAND_NORM_DIM);
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

function trimActionFrames(frames: Array<{ pose: number[]; hands: number[]; validMask: number[] }>) {
  if (!ENABLE_ACTION_TRIM || frames.length === 0) return frames;

  const leftMotion = handMotionScore(
    frames.map((frame) => frame.hands.slice(LEFT_HAND_NORM_START, LEFT_HAND_NORM_START + LEFT_HAND_NORM_DIM)),
    frames.map((frame) => frame.validMask[VALID_LEFT_HAND] === 1),
  );
  const rightMotion = handMotionScore(
    frames.map((frame) => frame.hands.slice(RIGHT_HAND_NORM_START, RIGHT_HAND_NORM_START + RIGHT_HAND_NORM_DIM)),
    frames.map((frame) => frame.validMask[VALID_RIGHT_HAND] === 1),
  );
  const poseMotion = poseMotionScore(
    frames.map((frame) => frame.pose.slice(0, POSE_NORM_DIM)),
    frames.map((frame) => frame.validMask[VALID_POSE] === 1),
  );

  const handPresent = frames.map((frame) => frame.validMask[VALID_LEFT_HAND] === 1 || frame.validMask[VALID_RIGHT_HAND] === 1);
  const handActive = frames.map((_, index) => leftMotion[index] > ACTION_TRIM_HAND_MOTION_THRESHOLD || rightMotion[index] > ACTION_TRIM_HAND_MOTION_THRESHOLD);
  const poseActive = frames.map((_, index) => poseMotion[index] > ACTION_TRIM_POSE_MOTION_THRESHOLD);
  const active = handActive.some(Boolean) ? handActive : handPresent.some(Boolean) ? handPresent : poseActive;

  if (!active.some(Boolean)) return frames;

  const activeIndices = active.map((value, index) => (value ? index : -1)).filter((index) => index >= 0);
  const [start, end] = expandBounds(
    activeIndices[0] - ACTION_TRIM_MARGIN_FRAMES,
    activeIndices[activeIndices.length - 1] + ACTION_TRIM_MARGIN_FRAMES + 1,
    frames.length,
    ACTION_TRIM_MIN_KEEP_FRAMES,
  );

  return end > start ? frames.slice(start, end) : frames;
}

function resampleFrames<T>(frames: T[], outputFrames: number) {
  if (frames.length === 0 || outputFrames <= 0 || frames.length === outputFrames) return frames;
  const indices = getSampleIndices(frames.length, outputFrames);
  return indices.map((index) => frames[index]);
}

function getSampleIndices(totalFrames: number, targetFrames: number) {
  if (totalFrames <= 1) return Array.from({ length: targetFrames }, () => 0);
  return Array.from({ length: targetFrames }, (_, index) => {
    const value = Math.round((index * (totalFrames - 1)) / Math.max(targetFrames - 1, 1));
    return Math.min(totalFrames - 1, Math.max(0, value));
  });
}

function handMotionScore(handBlocks: number[][], validMask: boolean[]) {
  const motion = createZeroArray(handBlocks.length);
  for (let frameIndex = 1; frameIndex < handBlocks.length; frameIndex += 1) {
    if (!validMask[frameIndex] || !validMask[frameIndex - 1]) continue;
    motion[frameIndex] = Math.max(motion[frameIndex], meanLandmarkDistance(handBlocks[frameIndex], handBlocks[frameIndex - 1], HAND_LANDMARK_COUNT, 3));
    motion[frameIndex - 1] = Math.max(motion[frameIndex - 1], motion[frameIndex]);
  }
  return motion;
}

function poseMotionScore(poseBlocks: number[][], validMask: boolean[]) {
  const motion = createZeroArray(poseBlocks.length);
  const upperBody = [11, 12, 13, 14, 15, 16];
  for (let frameIndex = 1; frameIndex < poseBlocks.length; frameIndex += 1) {
    if (!validMask[frameIndex] || !validMask[frameIndex - 1]) continue;
    let total = 0;
    for (const landmarkIndex of upperBody) {
      const base = landmarkIndex * POSE_VALUES_PER_LANDMARK;
      total += distance3d(poseBlocks[frameIndex], poseBlocks[frameIndex - 1], base);
    }
    const value = total / upperBody.length;
    motion[frameIndex] = Math.max(motion[frameIndex], value);
    motion[frameIndex - 1] = Math.max(motion[frameIndex - 1], value);
  }
  return motion;
}

function meanLandmarkDistance(current: number[], previous: number[], landmarkCount: number, valuesPerLandmark: number) {
  let total = 0;
  for (let index = 0; index < landmarkCount; index += 1) {
    total += distance3d(current, previous, index * valuesPerLandmark);
  }
  return total / landmarkCount;
}

function distance3d(current: number[], previous: number[], base: number) {
  const dx = (current[base] ?? 0) - (previous[base] ?? 0);
  const dy = (current[base + 1] ?? 0) - (previous[base + 1] ?? 0);
  const dz = (current[base + 2] ?? 0) - (previous[base + 2] ?? 0);
  return Math.hypot(dx, dy, dz);
}

function expandBounds(start: number, end: number, total: number, minKeep: number): [number, number] {
  let nextStart = Math.max(0, start);
  let nextEnd = Math.min(total, end);

  if (minKeep <= 0 || nextEnd - nextStart >= minKeep || total <= nextEnd - nextStart) {
    return [nextStart, nextEnd];
  }

  const missing = minKeep - (nextEnd - nextStart);
  const leftExtra = Math.floor(missing / 2);
  const rightExtra = missing - leftExtra;
  nextStart = Math.max(0, nextStart - leftExtra);
  nextEnd = Math.min(total, nextEnd + rightExtra);

  if (nextEnd - nextStart < minKeep) {
    if (nextStart === 0) nextEnd = Math.min(total, minKeep);
    else if (nextEnd === total) nextStart = Math.max(0, total - minKeep);
  }

  return [nextStart, nextEnd];
}

function normalizePoseSequence(pose: number[][]) {
  return pose.map((frame) => {
    const normalized = frame.slice();
    normalizePoseBlock(normalized, 0);
    normalizePoseBlock(normalized, POSE_NORM_DIM);
    return normalized;
  });
}

function normalizePoseBlock(frame: number[], start: number) {
  const neck = averageLandmark(frame, start, 11, 12, POSE_VALUES_PER_LANDMARK);
  const head = averageLandmark(frame, start, 7, 8, POSE_VALUES_PER_LANDMARK);
  const scale = Math.hypot(head[0] - neck[0], head[1] - neck[1], head[2] - neck[2]);
  const hasPose = hasNonZeroValues(frame.slice(start, start + POSE_NORM_DIM).filter((_, index) => index % POSE_VALUES_PER_LANDMARK < 3));

  if (!hasPose || scale <= EPSILON) return;

  for (let index = 0; index < POSE_LANDMARK_COUNT; index += 1) {
    const base = start + index * POSE_VALUES_PER_LANDMARK;
    frame[base] = (frame[base] - neck[0]) / scale;
    frame[base + 1] = (frame[base + 1] - neck[1]) / scale;
    frame[base + 2] = (frame[base + 2] - neck[2]) / scale;
  }
}

function averageLandmark(frame: number[], start: number, firstIndex: number, secondIndex: number, valuesPerLandmark: number): [number, number, number] {
  const first = start + firstIndex * valuesPerLandmark;
  const second = start + secondIndex * valuesPerLandmark;
  return [
    ((frame[first] ?? 0) + (frame[second] ?? 0)) * 0.5,
    ((frame[first + 1] ?? 0) + (frame[second + 1] ?? 0)) * 0.5,
    ((frame[first + 2] ?? 0) + (frame[second + 2] ?? 0)) * 0.5,
  ];
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
      handsOut.map((frame) => frame.slice(block.start, block.start + LEFT_HAND_NORM_DIM)),
      validMask.map((frame) => frame[block.validColumn] === 1),
    );

    for (let frameIndex = 0; frameIndex < handsOut.length; frameIndex += 1) {
      const normalized = normalizeHandBlock(interpolated[frameIndex]);
      for (let dim = 0; dim < LEFT_HAND_NORM_DIM; dim += 1) {
        previewHands[frameIndex][block.start + dim] = interpolated[frameIndex][dim] ?? 0;
        handsOut[frameIndex][block.start + dim] = normalized[dim] ?? 0;
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

function normalizeHandBlock(handBlock: number[]) {
  if (!hasNonZeroValues(handBlock)) return handBlock.slice();
  const wrist = [handBlock[0] ?? 0, handBlock[1] ?? 0, handBlock[2] ?? 0];
  const output = handBlock.slice();
  for (let index = 0; index < HAND_LANDMARK_COUNT; index += 1) {
    const base = index * HAND_VALUES_PER_LANDMARK;
    output[base] = (output[base] ?? 0) - wrist[0];
    output[base + 1] = (output[base + 1] ?? 0) - wrist[1];
    output[base + 2] = (output[base + 2] ?? 0) - wrist[2];
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
  const minorityRatio = Math.min(leftCount, rightCount) / Math.max(Math.max(leftCount, rightCount), 1);

  if (minorityRatio > HAND_SIDE_MINORITY_RATIO_THRESHOLD) return { hands: handsOut, validMask: validOut };

  const dominantSlices = getHandSlices(dominant);
  const minoritySlices = getHandSlices(minority);

  for (let frameIndex = 0; frameIndex < handsOut.length; frameIndex += 1) {
    const hasMinority = validOut[frameIndex][minoritySlices.validColumn] === 1;
    const hasDominant = validOut[frameIndex][dominantSlices.validColumn] === 1;

    if (hasMinority && !hasDominant) {
      copyRange(handsOut[frameIndex], minoritySlices.normStart, handsOut[frameIndex], dominantSlices.normStart, LEFT_HAND_NORM_DIM);
      copyRange(handsOut[frameIndex], minoritySlices.worldStart, handsOut[frameIndex], dominantSlices.worldStart, LEFT_HAND_WORLD_DIM);
      validOut[frameIndex][dominantSlices.validColumn] = 1;
    }

    if (hasMinority) {
      fillRange(handsOut[frameIndex], minoritySlices.normStart, LEFT_HAND_NORM_DIM, 0);
      fillRange(handsOut[frameIndex], minoritySlices.worldStart, LEFT_HAND_WORLD_DIM, 0);
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
