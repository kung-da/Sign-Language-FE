import { useCallback, useEffect, useRef, useState } from "react";

export function useCamera() {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setStream(null);
  }, []);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const nextStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
      });
      streamRef.current = nextStream;
      setStream(nextStream);
    } catch {
      setError("Camera permission was denied or no camera is available.");
    }
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  return {
    startCamera,
    stopCamera,
    stream,
    isCameraActive: Boolean(stream),
    error,
  };
}
