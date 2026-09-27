/*
 * Smart Cradle rolling-window cry-pattern ML adapter.
 *
 * This adapter receives a COMPLETE 3-second / 48,000-sample
 * PCM window from the Node rolling buffer.
 *
 * It does NOT send individual MQTT audio chunks to the model.
 *
 * Product wording:
 *   "probable cry pattern"
 *
 * This is not a medical diagnosis.
 */

async function analyze(window) {
  const url = process.env.CRY_MODEL_URL || 'http://127.0.0.1:8001/infer';

  const started = Date.now();

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json'
      },
      body: JSON.stringify({
        audio_base64: window.audio_base64,
        format: 'pcm_s16le_base64',
        sample_rate: window.sample_rate,
        device_id: window.device_id,
        event_id: window.last_event_id,
        window_id: window.window_id
      })
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(
        `model service HTTP ${response.status}: ${body}`
      );
    }

    const result = await response.json();

    return {
      audio_event_id: window.last_event_id || null,
      device_id: window.device_id,
      occurred_at: window.occurred_at || null,

      window_id: window.window_id,
      window_start_at: window.window_start_at || null,
      window_end_at: window.window_end_at || null,
      window_sample_count: window.sample_count,
      source_event_ids: window.source_event_ids || [],

      status: result.model_status || result.status || 'inference_complete',
      model_version: result.model_version || 'unknown',

      probable_pattern: result.prediction || result.probable_pattern || 'REVIEW',
      probability: Number.isFinite(result.confidence)
        ? result.confidence
        : null,

      probabilities: result.probabilities || {},
      decision: result.decision || 'REVIEW',
      needs_review: result.needs_review !== false,
      uncertainty_reasons: result.uncertainty_reasons || [],

      inference_ms: Date.now() - started,
      service_inference_ms: Number.isFinite(result.service_inference_ms)
        ? result.service_inference_ms
        : null,

      features: result.feature || {},

      error: null
    };
  } catch (error) {
    return {
      audio_event_id: window.last_event_id || null,
      device_id: window.device_id,
      occurred_at: window.occurred_at || null,

      window_id: window.window_id,
      window_start_at: window.window_start_at || null,
      window_end_at: window.window_end_at || null,
      window_sample_count: window.sample_count,
      source_event_ids: window.source_event_ids || [],

      status: 'inference_error',
      model_version: 'external-cry-model',

      probable_pattern: 'REVIEW',
      probability: null,

      probabilities: {},
      decision: 'REVIEW',
      needs_review: true,
      uncertainty_reasons: ['inference_error'],

      inference_ms: Date.now() - started,
      service_inference_ms: null,

      features: {},
      error: error.message
    };
  }
}

module.exports = { analyze };
