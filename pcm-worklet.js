// Captures the mic continuously and emits 20ms frames of 16kHz mono Int16 PCM —
// exactly the frame size webrtcvad expects server-side, and small enough for
// low-latency real-time streaming and barge-in detection.
class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameSize = 320; // 20ms @ 16kHz
    this.buffer = new Float32Array(this.frameSize);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channel = input[0];
      for (let i = 0; i < channel.length; i++) {
        this.buffer[this.offset++] = channel[i];
        if (this.offset === this.frameSize) {
          const int16 = new Int16Array(this.frameSize);
          for (let j = 0; j < this.frameSize; j++) {
            const s = Math.max(-1, Math.min(1, this.buffer[j]));
            int16[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          this.port.postMessage(int16.buffer, [int16.buffer]);
          this.offset = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PCMCaptureProcessor);
