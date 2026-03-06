// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! Energy-based VAD, pause detection, and turn-taking logic.
//! This supplements the model's built-in neural VAD head with
//! signal-level heuristics for pause/boundary/turn-end detection.

/// Configuration for energy-based pause and turn-taking detection.
#[derive(Debug, Clone)]
pub struct PauseDetectorConfig {
    pub energy_threshold: f64,
    pub min_pause_ms: u64,
    pub min_boundary_ms: u64,
    pub turn_silence_ms: u64,
    pub min_turn_speech_ms: u64,
    pub speech_hysteresis_ms: u64,
    pub silence_hysteresis_ms: u64,
}

impl Default for PauseDetectorConfig {
    fn default() -> Self {
        Self {
            energy_threshold: 0.005,
            min_pause_ms: 100,
            min_boundary_ms: 200,
            turn_silence_ms: 500,
            min_turn_speech_ms: 200,
            speech_hysteresis_ms: 60,
            silence_hysteresis_ms: 150,
        }
    }
}

/// Events emitted by the pause/turn detector.
#[derive(Debug, Clone)]
pub enum PauseEvent {
    SpeechStart { timestamp_ns: u64 },
    SpeechEnd { timestamp_ns: u64, duration_ns: u64 },
    PauseDetected { timestamp_ns: u64, duration_ns: u64, energy: f64 },
    BoundaryDetected { timestamp_ns: u64, confidence: f64 },
    TurnEnd { timestamp_ns: u64, speech_duration_ns: u64, silence_ns: u64 },
    SilenceDetected { timestamp_ns: u64, duration_ns: u64 },
}

pub struct PauseDetector {
    config: PauseDetectorConfig,
    /// Whether we currently consider speech active (debounced).
    is_speech: bool,
    /// Timestamp (ns) when current speech started.
    speech_start_ns: u64,
    /// Timestamp (ns) when current silence started.
    silence_start_ns: u64,
    /// Running position in nanoseconds.
    position_ns: u64,
    /// Accumulator for hysteresis (consecutive speech frames in ms).
    consecutive_speech_ms: u64,
    /// Accumulator for hysteresis (consecutive silence frames in ms).
    consecutive_silence_ms: u64,
}

impl PauseDetector {
    pub fn new(config: PauseDetectorConfig) -> Self {
        Self {
            config,
            is_speech: false,
            speech_start_ns: 0,
            silence_start_ns: 0,
            position_ns: 0,
            consecutive_speech_ms: 0,
            consecutive_silence_ms: 0,
        }
    }

    /// Process a chunk of audio samples and return any detected events.
    /// `sample_rate` is needed to convert sample counts to time.
    pub fn process(&mut self, samples: &[f32], sample_rate: u32) -> Vec<PauseEvent> {
        let mut events = Vec::new();
        let chunk_duration_ns =
            (samples.len() as u64 * 1_000_000_000) / sample_rate as u64;
        let chunk_duration_ms = chunk_duration_ns / 1_000_000;

        // Compute RMS energy.
        let energy = if samples.is_empty() {
            0.0
        } else {
            let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
            (sum_sq / samples.len() as f64).sqrt()
        };

        let frame_has_speech = energy > self.config.energy_threshold;

        if frame_has_speech {
            self.consecutive_speech_ms += chunk_duration_ms;
            self.consecutive_silence_ms = 0;
        } else {
            self.consecutive_silence_ms += chunk_duration_ms;
            self.consecutive_speech_ms = 0;
        }

        // Transition: silence -> speech (debounced).
        if !self.is_speech
            && self.consecutive_speech_ms >= self.config.speech_hysteresis_ms
        {
            self.is_speech = true;
            self.speech_start_ns = self.position_ns;
            events.push(PauseEvent::SpeechStart {
                timestamp_ns: self.position_ns,
            });
        }

        // Transition: speech -> silence (debounced).
        if self.is_speech
            && self.consecutive_silence_ms >= self.config.silence_hysteresis_ms
        {
            let speech_dur = self.position_ns - self.speech_start_ns;
            self.is_speech = false;
            self.silence_start_ns = self.position_ns
                - self.consecutive_silence_ms * 1_000_000;

            events.push(PauseEvent::SpeechEnd {
                timestamp_ns: self.position_ns,
                duration_ns: speech_dur,
            });

            // Check for pause.
            if self.consecutive_silence_ms >= self.config.min_pause_ms {
                let silence_dur = self.consecutive_silence_ms * 1_000_000;
                events.push(PauseEvent::PauseDetected {
                    timestamp_ns: self.position_ns,
                    duration_ns: silence_dur,
                    energy,
                });
            }

            // Check for boundary.
            if self.consecutive_silence_ms >= self.config.min_boundary_ms {
                let confidence = (self.consecutive_silence_ms as f64
                    / self.config.turn_silence_ms as f64)
                    .min(1.0);
                events.push(PauseEvent::BoundaryDetected {
                    timestamp_ns: self.position_ns,
                    confidence,
                });
            }
        }

        // Turn-end detection: long silence after sufficient speech.
        if !self.is_speech
            && self.consecutive_silence_ms >= self.config.turn_silence_ms
        {
            let speech_dur = if self.speech_start_ns < self.silence_start_ns {
                self.silence_start_ns - self.speech_start_ns
            } else {
                0
            };
            let speech_dur_ms = speech_dur / 1_000_000;
            if speech_dur_ms >= self.config.min_turn_speech_ms {
                let silence_ns = self.consecutive_silence_ms * 1_000_000;
                events.push(PauseEvent::TurnEnd {
                    timestamp_ns: self.position_ns,
                    speech_duration_ns: speech_dur,
                    silence_ns,
                });
                // Reset to avoid repeated turn-end signals.
                self.speech_start_ns = self.position_ns;
            }
        }

        // General silence notification.
        if !self.is_speech && self.consecutive_silence_ms >= self.config.turn_silence_ms {
            let silence_ns = self.consecutive_silence_ms * 1_000_000;
            events.push(PauseEvent::SilenceDetected {
                timestamp_ns: self.position_ns,
                duration_ns: silence_ns,
            });
        }

        self.position_ns += chunk_duration_ns;
        events
    }
}
