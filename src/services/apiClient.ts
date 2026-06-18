const BASE_URL = "http://127.0.0.1:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`API ${response.status}: ${body || response.statusText}`);
  }

  return response.json() as Promise<T>;
}

export interface HealthResponse {
  status: string;
  checkpoint_exists: boolean;
  torch_available: boolean;
}

export interface PredictResponseItem {
  class_index: number;
  label: string;
  probability: number;
}

export interface PredictResponse {
  top_k: PredictResponseItem[][];
  batch_size: number;
  model_inference_ms: number;
  end_to_end_ms: number;
}

export interface MetadataResponse {
  baseline_dir: string;
  checkpoint_exists: boolean;
  checkpoint_size_mb: number | null;
  config: Record<string, unknown>;
  latency: Record<string, number>;
  history_summary: {
    best_epoch_by_val_acc: number | null;
    best_val_acc: number | null;
    epochs_recorded: number | null;
  };
  runtime: {
    torch_available: boolean;
    normalization_available: boolean;
    labels_available: boolean;
  };
}

export interface BenchmarkResponse {
  runs: number;
  warmup: number;
  mean_ms: number;
  median_ms: number;
  p95_ms: number;
  min_ms: number;
  max_ms: number;
}

export const apiClient = {
  fetchHealth(): Promise<HealthResponse> {
    return request<HealthResponse>("/health");
  },

  fetchMetadata(): Promise<MetadataResponse> {
    return request<MetadataResponse>("/model/metadata");
  },

  postPredict(sequence: number[][], topK = 5): Promise<PredictResponse> {
    return request<PredictResponse>("/model/predict", {
      method: "POST",
      body: JSON.stringify({ sequence, top_k: topK }),
    });
  },

  postBenchmark(runs = 30, warmup = 5): Promise<BenchmarkResponse> {
    return request<BenchmarkResponse>("/model/benchmark", {
      method: "POST",
      body: JSON.stringify({ runs, warmup }),
    });
  },
};
