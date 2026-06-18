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
      streamRef.current?.getTracks().forEach((track) => track.stop());

      const nextStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
          frameRate: { ideal: 60, max: 60 },
        },
        audio: false,
      });
      streamRef.current = nextStream;
      setStream(nextStream);
    } catch (cameraError) {
      const message =
        cameraError instanceof DOMException && cameraError.name === "NotAllowedError"
          ? "Camera permission was denied. Please allow camera access and try again."
          : "No camera is available. Try another device or browser.";

      setError(message);
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
