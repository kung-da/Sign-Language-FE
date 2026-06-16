import { Upload, Video, VideoOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "../ui/Button";
import { GlassCard } from "../ui/GlassCard";

const dots = [
  ["28%", "28%"],
  ["38%", "34%"],
  ["50%", "30%"],
  ["58%", "42%"],
  ["43%", "50%"],
  ["36%", "62%"],
  ["55%", "64%"],
  ["64%", "55%"],
  ["47%", "73%"],
];

interface CameraPreviewProps {
  stream: MediaStream | null;
  isActive: boolean;
  onStart: () => void;
  onStop: () => void;
  error?: string | null;
}

export function CameraPreview({ stream, isActive, onStart, onStop, error }: CameraPreviewProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  return (
    <GlassCard className="overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 p-4">
        <div>
          <p className="font-semibold text-text">Webcam / Video Input</p>
          <p className="text-sm text-muted">Local browser demo. No frames are uploaded.</p>
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
                if (file) setUploadedVideo(URL.createObjectURL(file));
              }}
            />
          </label>
          <Button variant="danger" icon={<VideoOff size={18} />} onClick={onStop} disabled={!isActive}>
            Stop
          </Button>
        </div>
      </div>
      <div className="relative aspect-video bg-slate-950">
        {isActive ? (
          <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover opacity-80" />
        ) : uploadedVideo ? (
          <video src={uploadedVideo} controls className="h-full w-full object-cover opacity-80" />
        ) : (
          <div className="grid h-full place-items-center px-6 text-center">
            <div>
              <div className="mx-auto mb-4 grid h-16 w-16 place-items-center rounded-full border border-white/10 bg-white/10">
                <Video size={28} className="text-cyan" />
              </div>
              <p className="font-semibold text-text">Camera preview is idle</p>
              <p className="mt-2 max-w-md text-sm text-muted">
                Start the camera or upload a local video to simulate realtime recognition.
              </p>
              {error && <p className="mt-3 text-sm text-danger">{error}</p>}
            </div>
          </div>
        )}
        {(isActive || uploadedVideo) && (
          <>
            <div className="scan-line" />
            {dots.map(([left, top], index) => (
              <span key={`${left}-${top}`} className="keypoint-dot" style={{ left, top, animationDelay: `${index * 80}ms` }} />
            ))}
          </>
        )}
      </div>
    </GlassCard>
  );
}
