'use strict';

/*
 * Live MQTT rolling audio buffer.
 *
 * Input:
 *   Arbitrary PCM S16LE mono chunks.
 *
 * Output:
 *   48,000-sample / 3-second windows every 16,000 samples.
 *
 * MQTT packet boundaries do not affect inference windows.
 */

const WINDOW_SAMPLES = 48000;
const HOP_SAMPLES = 16000;
const SAMPLE_RATE = 16000;

class RollingAudioBuffer {
  constructor(deviceId) {
    this.deviceId = deviceId;
    this.samples = Buffer.alloc(0);
    this.events = [];
    this.windowIndex = 0;
    this.totalSamplesReceived = 0;
  }

  addChunk({
    pcmBuffer,
    eventId,
    occurredAt
  }) {
    if (!Buffer.isBuffer(pcmBuffer)) {
      throw new TypeError('pcmBuffer must be a Buffer');
    }

    if (pcmBuffer.length % 2 !== 0) {
      throw new Error('PCM S16LE buffer has an odd byte length');
    }

    if (pcmBuffer.length === 0) {
      return [];
    }

    const sampleCount = pcmBuffer.length / 2;

    this.events.push({
      eventId: eventId || null,
      occurredAt: occurredAt || null,
      sampleCount
    });

    this.samples = Buffer.concat([
      this.samples,
      pcmBuffer
    ]);

    this.totalSamplesReceived += sampleCount;

    const windows = [];

    while (this.samples.length >= WINDOW_SAMPLES * 2) {
      const windowBuffer = Buffer.from(
        this.samples.subarray(0, WINDOW_SAMPLES * 2)
      );

      const sourceEventIds = this.events
        .map(event => event.eventId)
        .filter(Boolean);

      const firstEvent = this.events[0] || {};
      const lastEvent = this.events[this.events.length - 1] || {};

      this.windowIndex += 1;

      const windowStartAt = firstEvent.occurredAt || null;

      let windowEndAt = null;

      if (windowStartAt) {
        const startMs = Date.parse(windowStartAt);

        if (Number.isFinite(startMs)) {
          windowEndAt = new Date(
            startMs + 3000
          ).toISOString();
        }
      }

      windows.push({
        windowId:
          `${this.deviceId}-rolling-${this.windowIndex}`,

        deviceId: this.deviceId,

        sampleRate: SAMPLE_RATE,
        sampleCount: WINDOW_SAMPLES,

        audioBuffer: windowBuffer,

        sourceEventIds,

        firstEventId: firstEvent.eventId || null,
        lastEventId: lastEvent.eventId || null,

        windowStartAt,
        windowEndAt
      });

      /*
       * Advance exactly one second.
       *
       * Keep the remaining 2 seconds for overlap with
       * the next inference window.
       */
      const hopBytes = HOP_SAMPLES * 2;

      this.samples = this.samples.subarray(hopBytes);

      this.consumeEventSamples(HOP_SAMPLES);
    }

    return windows;
  }

  consumeEventSamples(sampleCount) {
    let remaining = sampleCount;

    while (remaining > 0 && this.events.length > 0) {
      const event = this.events[0];

      if (event.sampleCount <= remaining) {
        remaining -= event.sampleCount;
        this.events.shift();
      } else {
        event.sampleCount -= remaining;
        remaining = 0;
      }
    }
  }

  clear() {
    this.samples = Buffer.alloc(0);
    this.events = [];
  }

  getBufferedSamples() {
    return this.samples.length / 2;
  }
}

module.exports = {
  RollingAudioBuffer,
  WINDOW_SAMPLES,
  HOP_SAMPLES,
  SAMPLE_RATE
};
