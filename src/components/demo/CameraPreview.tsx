import { Maximize, Minimize, Upload, Video, VideoOff } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { CameraSettings } from "../../hooks/useCamera";
import { useMediaPipeLandmarks, type WorkerLandmarks } from "../../hooks/useMediaPipeLandmarks";
import { PerformancePanel } from "./PerformancePanel";
import { Button } from "../ui/Button";
import { GlassCard } from "../ui/GlassCard";

interface CameraPreviewProps {
  stream: MediaStream | null;
  cameraSettings: CameraSettings | null;
  isActive: boolean;
  onStart: () => void;
  onStop: () => void;
  error?: string | null;
  onLandmarks?: (landmarks: WorkerLandmarks) => void;
  onVideoUpload?: (file: File) => Promise<void>;
  onVideoActiveChange?: (isActive: boolean) => void;
}

export function CameraPreview({ stream, cameraSettings, isActive, onStart, onStop, error, onLandmarks, onVideoUpload, onVideoActiveChange }: CameraPreviewProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const hasVideoPreview = isActive || !!uploadedVideo;

  // Keep state in sync when user exits fullscreen via Escape key
  useEffect(() => {
    const handleChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", handleChange);
    return () => document.removeEventListener("fullscreenchange", handleChange);
  }, []);

  const toggleFullscreen = useCallback(async () => {
    if (!containerRef.current) return;
    try {
      if (!document.fullscreenElement) {
        await containerRef.current.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.warn("Fullscreen not supported", err);
    }
  }, []);
  const {
    counts,
    delegate,
    error: landmarkError,
    metrics,
    status: landmarkStatus,
  } = useMediaPipeLandmarks({
    videoRef,
    canvasRef,
    isActive,
    onLandmarks,
  });

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (uploadedVideo) {
      video.srcObject = null;
      return;
    }

    if (stream) {
      video.srcObject = stream;
    }
  }, [stream, uploadedVideo]);

  useEffect(() => {
    onVideoActiveChange?.(isActive);
  }, [isActive, onVideoActiveChange]);

  useEffect(() => {
    return () => {
      if (uploadedVideo) URL.revokeObjectURL(uploadedVideo);
    };
  }, [uploadedVideo]);


  return (
    <div className="space-y-4">
      <GlassCard className="overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 p-4">
        <div>
          <p className="font-semibold text-text">Webcam / Video Input</p>
          <p className="text-sm text-muted">Local MediaPipe landmarks. No frames are uploaded.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button icon={<Video size={18} />} onClick={onStart} disabled={isActive}>
            Start Camera
          </Button>
          <label className="inline-flex min-h-11 cursor-pointer items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/10 px-4 py-2 text-sm font-semibold text-text transition hover:bg-white/15">
            <Upload size={18} />
            Upload Video
            <input
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (!file) return;
                onStop();
                setUploadedVideo((currentUrl) => {
                  if (currentUrl) URL.revokeObjectURL(currentUrl);
                  return URL.createObjectURL(file);
                });
                void onVideoUpload?.(file);
              }}
            />
          </label>
          <Button
            variant="danger"
            icon={<VideoOff size={18} />}
            onClick={() => {
              setUploadedVideo((currentUrl) => {
                if (currentUrl) URL.revokeObjectURL(currentUrl);
                return null;
              });
              onStop();
            }}
            disabled={!hasVideoPreview}
          >
            Stop
          </Button>
        </div>
      </div>
      <div ref={containerRef} className={`relative bg-slate-950 ${isFullscreen ? "flex items-center justify-center h-screen" : "aspect-video"}`}>
        {hasVideoPreview ? (
          <>
            <video
              ref={videoRef}
              src={uploadedVideo ?? undefined}
              autoPlay
              controls={!!uploadedVideo}
              loop={!!uploadedVideo}
              playsInline
              muted
              onLoadedMetadata={() => {
                if (uploadedVideo) {
                  if (videoRef.current) videoRef.current.currentTime = 0;
                  void videoRef.current?.play().catch(() => undefined);
                }
              }}
              className="h-full w-full object-cover opacity-80"
            />
            <canvas
              ref={canvasRef}
              className="pointer-events-none absolute inset-0 h-full w-full object-cover"
            />
            <div className="absolute left-3 top-3 rounded-md border border-white/10 bg-slate-950/70 px-3 py-2 text-xs font-semibold text-text backdrop-blur">
              {landmarkStatus === "loading"
                ? "Loading MediaPipe..."
                : uploadedVideo
                  ? "Backend demo upload"
                  : landmarkStatus === "ready"
                  ? `${delegate ?? "CPU"} | Hands: ${counts.hands} | Face: ${counts.face} | Pose: ${counts.pose}`
                  : landmarkStatus === "error"
                    ? "MediaPipe unavailable"
                    : "Camera ready"}
            </div>
            <button
              onClick={toggleFullscreen}
              className="absolute right-3 top-3 grid h-9 w-9 place-items-center rounded-md border border-white/10 bg-slate-950/70 text-text backdrop-blur transition hover:bg-white/20"
              title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            >
              {isFullscreen ? <Minimize size={18} /> : <Maximize size={18} />}
            </button>

          </>
        ) : (
          <div className="grid h-full place-items-center px-6 text-center">
            <div>
              <div className="mx-auto mb-4 grid h-16 w-16 place-items-center rounded-full border border-white/10 bg-white/10">
                <Video size={28} className="text-cyan" />
              </div>
              <p className="font-semibold text-text">Camera preview is idle</p>
              <p className="mt-2 max-w-md text-sm text-muted">
                Start the camera or upload a local video to begin realtime recognition.
              </p>
              {error && <p className="mt-3 text-sm text-danger">{error}</p>}
              {landmarkError && <p className="mt-3 text-sm text-danger">{landmarkError}</p>}
            </div>
          </div>
        )}
        {hasVideoPreview && (
          <div aria-hidden="true">
            <div className="scan-line" />
          </div>
        )}
        {isActive && landmarkError && (
          <p className="absolute bottom-3 left-3 right-3 rounded-md border border-danger/30 bg-slate-950/80 px-3 py-2 text-sm text-danger">
            {landmarkError}
          </p>
        )}
      </div>
      </GlassCard>
      <PerformancePanel metrics={metrics} delegate={delegate} cameraSettings={cameraSettings} />
    </div>
  );
}
