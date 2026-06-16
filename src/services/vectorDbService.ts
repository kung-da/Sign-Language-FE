import type { NewSignPayload, TopPrediction } from "../types/recognition";

export interface VectorSearchResult {
  nearest: TopPrediction[];
  embedding: number[];
}

const createMockEmbedding = () =>
  Array.from({ length: 16 }, () => Number((Math.random() * 2 - 1).toFixed(4)));

export const vectorDbService = {
  async searchNearestVector(embedding = createMockEmbedding()): Promise<VectorSearchResult> {
    return {
      embedding,
      nearest: [
        { label: "Xin chao", gloss: "Greeting", confidence: 0.78 },
        { label: "Cam on", gloss: "Thanks", confidence: 0.14 },
        { label: "Tam biet", gloss: "Goodbye", confidence: 0.08 },
      ],
    };
  },

  async saveNewSign(payload: Omit<NewSignPayload, "embedding" | "createdAt" | "source">) {
    const savedPayload: NewSignPayload = {
      ...payload,
      embedding: createMockEmbedding(),
      source: "user-demo",
      createdAt: new Date().toISOString(),
    };

    return { ok: true, payload: savedPayload };
  },
};
