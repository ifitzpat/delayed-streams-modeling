// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! GStreamer element implementation for Kyutai STT.
//!
//! Pad-compatible with gst-whispertranscribe: accepts audio/x-raw on sink,
//! emits application/x-json-transcription,format=whisper on src, and
//! optionally accepts application/x-whisper-control on a request ctrl pad.

use std::sync::Mutex;

use glib::subclass::Signal;
use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;
use once_cell::sync::Lazy;

use crate::model::{ModelInfo, SttEvent, SttModel};
use crate::vad::{PauseDetector, PauseDetectorConfig, PauseEvent};

// ── Constants ──────────────────────────────────────────────────────────────

/// The Kyutai STT model expects 24 kHz mono audio.
const MODEL_SAMPLE_RATE: u32 = 24_000;

/// Native processing chunk: 1920 samples = 80 ms at 24 kHz.
const CHUNK_SAMPLES: usize = 1920;

// ── Internal state ─────────────────────────────────────────────────────────

struct State {
    model: SttModel,
    #[allow(dead_code)]
    info: ModelInfo,
    pause_detector: PauseDetector,
    /// Accumulator for incoming PCM until we have a full chunk.
    pcm_buffer: Vec<f32>,
    /// Segment index counter.
    segment_idx: u64,
    /// Running sample position (at MODEL_SAMPLE_RATE) for timestamp calculation.
    sample_position: u64,
    /// Resampler state, if the input rate differs from MODEL_SAMPLE_RATE.
    resampler: Option<rubato::FftFixedIn<f32>>,
    /// Input sample rate for resampler.
    input_rate: u32,
    /// Input channels.
    input_channels: u32,
}

#[derive(Default)]
pub struct KyutaiStt {
    state: Mutex<Option<State>>,
    props: Mutex<Properties>,
}

// ── Properties ─────────────────────────────────────────────────────────────

struct Properties {
    // Model & core
    hf_repo: String,
    model_path: String,
    language: String,
    use_cpu: bool,
    temperature: f32,
    initial_prompt: Option<String>,

    // Transcription
    detect_language: bool,

    // Timestamps
    use_unix_timestamps: bool,

    // VAD (model-level)
    enable_vad: bool,
    vad_threshold: f32,
    no_speech_threshold: f32,

    // Pause & turn-taking
    enable_pause_detection: bool,
    pause_energy_threshold: f64,
    pause_min_duration: u32,
    boundary_min_duration: u32,
    turn_silence_threshold: u32,
    min_turn_speech_duration: u32,
    signal_speech_hysteresis: u32,
    signal_silence_hysteresis: u32,
}

impl Default for Properties {
    fn default() -> Self {
        Self {
            hf_repo: "kyutai/stt-1b-en_fr-candle".to_string(),
            model_path: "model.safetensors".to_string(),
            language: "auto".to_string(),
            use_cpu: false,
            temperature: 0.0,
            initial_prompt: None,
            detect_language: true,
            use_unix_timestamps: false,
            enable_vad: true,
            vad_threshold: 0.7,
            no_speech_threshold: 0.6,
            enable_pause_detection: true,
            pause_energy_threshold: 0.005,
            pause_min_duration: 100,
            boundary_min_duration: 200,
            turn_silence_threshold: 500,
            min_turn_speech_duration: 200,
            signal_speech_hysteresis: 60,
            signal_silence_hysteresis: 150,
        }
    }
}

// ── Pad templates ──────────────────────────────────────────────────────────

static SINK_CAPS: Lazy<gst::Caps> = Lazy::new(|| {
    gst::Caps::builder("audio/x-raw")
        .field("format", gst::List::new(["F32LE", "S16LE"]))
        .field(
            "rate",
            gst::List::new([8000i32, 16000, 22050, 24000, 44100, 48000]),
        )
        .field("channels", gst::List::new([1i32, 2]))
        .build()
});

static SRC_CAPS: Lazy<gst::Caps> = Lazy::new(|| {
    gst::Caps::builder("application/x-json-transcription")
        .field("format", "whisper")
        .build()
});

static CTRL_CAPS: Lazy<gst::Caps> = Lazy::new(|| {
    gst::Caps::builder("application/x-whisper-control").build()
});

// ── Signals ────────────────────────────────────────────────────────────────

static SIGNALS: Lazy<Vec<Signal>> = Lazy::new(|| {
    vec![
        // Model lifecycle
        Signal::builder("model-loaded")
            .param_types([String::static_type()])
            .build(),
        Signal::builder("model-unloaded").build(),
        Signal::builder("model-load-failed")
            .param_types([String::static_type()])
            .build(),
        Signal::builder("model-info")
            .param_types([gst::Structure::static_type()])
            .build(),
        // Transcription events
        Signal::builder("segment-transcribed")
            .param_types([gst::Structure::static_type()])
            .build(),
        Signal::builder("language-detected")
            .param_types([String::static_type(), f64::static_type()])
            .build(),
        Signal::builder("transcription-started")
            .param_types([i64::static_type()])
            .build(),
        Signal::builder("transcription-completed")
            .param_types([i64::static_type(), i64::static_type()])
            .build(),
        // VAD & speech activity
        Signal::builder("vad-speech-detected")
            .param_types([i64::static_type(), bool::static_type()])
            .build(),
        Signal::builder("speech-start")
            .param_types([i64::static_type()])
            .build(),
        Signal::builder("speech-end")
            .param_types([i64::static_type(), i64::static_type()])
            .build(),
        Signal::builder("silence-detected")
            .param_types([i64::static_type(), i64::static_type()])
            .build(),
        Signal::builder("pause-detected")
            .param_types([i64::static_type(), i64::static_type(), f64::static_type()])
            .build(),
        Signal::builder("boundary-detected")
            .param_types([i64::static_type(), f64::static_type()])
            .build(),
        // Turn-taking
        Signal::builder("turn-end")
            .param_types([
                i64::static_type(),
                i64::static_type(),
                i64::static_type(),
            ])
            .build(),
        // Diagnostics
        Signal::builder("mode-changed")
            .param_types([i32::static_type(), i32::static_type()])
            .build(),
        Signal::builder("buffer-overflow")
            .param_types([u64::static_type()])
            .build(),
    ]
});

// ── Property IDs ───────────────────────────────────────────────────────────

const PROP_HF_REPO: &str = "model";
const PROP_MODEL_PATH: &str = "model-path";
const PROP_LANGUAGE: &str = "language";
const PROP_USE_CPU: &str = "use-cpu";
const PROP_TEMPERATURE: &str = "temperature";
const PROP_INITIAL_PROMPT: &str = "initial-prompt";
const PROP_DETECT_LANGUAGE: &str = "detect-language";
const PROP_USE_UNIX_TS: &str = "use-unix-timestamps";
const PROP_ENABLE_VAD: &str = "enable-vad";
const PROP_VAD_THRESHOLD: &str = "vad-threshold";
const PROP_NO_SPEECH_THRESHOLD: &str = "no-speech-threshold";
const PROP_ENABLE_PAUSE: &str = "enable-pause-detection";
const PROP_PAUSE_ENERGY: &str = "pause-energy-threshold";
const PROP_PAUSE_MIN_DUR: &str = "pause-min-duration";
const PROP_BOUNDARY_MIN_DUR: &str = "boundary-min-duration";
const PROP_TURN_SILENCE: &str = "turn-silence-threshold";
const PROP_MIN_TURN_SPEECH: &str = "min-turn-speech-duration";
const PROP_SPEECH_HYST: &str = "signal-speech-hysteresis";
const PROP_SILENCE_HYST: &str = "signal-silence-hysteresis";

// ── GObject / GstElement impl ──────────────────────────────────────────────

#[glib::object_subclass]
impl ObjectSubclass for KyutaiStt {
    const NAME: &'static str = "KyutaiStt";
    type Type = super::KyutaiStt;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for KyutaiStt {
    fn signals() -> &'static [Signal] {
        SIGNALS.as_ref()
    }

    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                // Model & core
                glib::ParamSpecString::builder(PROP_HF_REPO)
                    .nick("Model")
                    .blurb("HuggingFace repo ID or path to model")
                    .default_value(Some("kyutai/stt-1b-en_fr-candle"))
                    .build(),
                glib::ParamSpecString::builder(PROP_MODEL_PATH)
                    .nick("Model path")
                    .blurb("Filename of the model weights within the repo")
                    .default_value(Some("model.safetensors"))
                    .build(),
                glib::ParamSpecString::builder(PROP_LANGUAGE)
                    .nick("Language")
                    .blurb("Language code (ISO 639-1) or 'auto'")
                    .default_value(Some("auto"))
                    .build(),
                glib::ParamSpecBoolean::builder(PROP_USE_CPU)
                    .nick("Use CPU")
                    .blurb("Force CPU inference (disable GPU)")
                    .default_value(false)
                    .build(),
                glib::ParamSpecFloat::builder(PROP_TEMPERATURE)
                    .nick("Temperature")
                    .blurb("Sampling temperature")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(0.0)
                    .build(),
                glib::ParamSpecString::builder(PROP_INITIAL_PROMPT)
                    .nick("Initial prompt")
                    .blurb("Optional context prompt")
                    .build(),
                // Transcription
                glib::ParamSpecBoolean::builder(PROP_DETECT_LANGUAGE)
                    .nick("Detect language")
                    .blurb("Auto-detect language")
                    .default_value(true)
                    .build(),
                glib::ParamSpecBoolean::builder(PROP_USE_UNIX_TS)
                    .nick("Unix timestamps")
                    .blurb("Use absolute unix timestamps in output")
                    .default_value(false)
                    .build(),
                // VAD
                glib::ParamSpecBoolean::builder(PROP_ENABLE_VAD)
                    .nick("Enable VAD")
                    .blurb("Enable voice activity detection")
                    .default_value(true)
                    .build(),
                glib::ParamSpecFloat::builder(PROP_VAD_THRESHOLD)
                    .nick("VAD threshold")
                    .blurb("VAD speech probability threshold")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(0.7)
                    .build(),
                glib::ParamSpecFloat::builder(PROP_NO_SPEECH_THRESHOLD)
                    .nick("No-speech threshold")
                    .blurb("No-speech probability threshold")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(0.6)
                    .build(),
                // Pause & turn-taking
                glib::ParamSpecBoolean::builder(PROP_ENABLE_PAUSE)
                    .nick("Enable pause detection")
                    .blurb("Enable energy-based pause signals")
                    .default_value(true)
                    .build(),
                glib::ParamSpecDouble::builder(PROP_PAUSE_ENERGY)
                    .nick("Pause energy threshold")
                    .blurb("RMS energy threshold for pause detection")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(0.005)
                    .build(),
                glib::ParamSpecUInt::builder(PROP_PAUSE_MIN_DUR)
                    .nick("Min pause duration")
                    .blurb("Minimum pause duration in ms")
                    .minimum(0)
                    .maximum(5000)
                    .default_value(100)
                    .build(),
                glib::ParamSpecUInt::builder(PROP_BOUNDARY_MIN_DUR)
                    .nick("Min boundary duration")
                    .blurb("Minimum pause for boundary detection in ms")
                    .minimum(0)
                    .maximum(5000)
                    .default_value(200)
                    .build(),
                glib::ParamSpecUInt::builder(PROP_TURN_SILENCE)
                    .nick("Turn silence threshold")
                    .blurb("Silence duration to signal turn-end in ms")
                    .minimum(100)
                    .maximum(5000)
                    .default_value(500)
                    .build(),
                glib::ParamSpecUInt::builder(PROP_MIN_TURN_SPEECH)
                    .nick("Min turn speech")
                    .blurb("Minimum speech before turn-end in ms")
                    .minimum(0)
                    .maximum(5000)
                    .default_value(200)
                    .build(),
                glib::ParamSpecUInt::builder(PROP_SPEECH_HYST)
                    .nick("Speech hysteresis")
                    .blurb("Debounce: consecutive speech before start (ms)")
                    .minimum(0)
                    .maximum(1000)
                    .default_value(60)
                    .build(),
                glib::ParamSpecUInt::builder(PROP_SILENCE_HYST)
                    .nick("Silence hysteresis")
                    .blurb("Debounce: consecutive silence before end (ms)")
                    .minimum(0)
                    .maximum(1000)
                    .default_value(150)
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut props = self.props.lock().unwrap();
        match pspec.name() {
            PROP_HF_REPO => props.hf_repo = value.get().unwrap(),
            PROP_MODEL_PATH => props.model_path = value.get().unwrap(),
            PROP_LANGUAGE => props.language = value.get().unwrap(),
            PROP_USE_CPU => props.use_cpu = value.get().unwrap(),
            PROP_TEMPERATURE => props.temperature = value.get().unwrap(),
            PROP_INITIAL_PROMPT => props.initial_prompt = value.get().ok(),
            PROP_DETECT_LANGUAGE => props.detect_language = value.get().unwrap(),
            PROP_USE_UNIX_TS => props.use_unix_timestamps = value.get().unwrap(),
            PROP_ENABLE_VAD => props.enable_vad = value.get().unwrap(),
            PROP_VAD_THRESHOLD => props.vad_threshold = value.get().unwrap(),
            PROP_NO_SPEECH_THRESHOLD => props.no_speech_threshold = value.get().unwrap(),
            PROP_ENABLE_PAUSE => props.enable_pause_detection = value.get().unwrap(),
            PROP_PAUSE_ENERGY => props.pause_energy_threshold = value.get().unwrap(),
            PROP_PAUSE_MIN_DUR => props.pause_min_duration = value.get().unwrap(),
            PROP_BOUNDARY_MIN_DUR => props.boundary_min_duration = value.get().unwrap(),
            PROP_TURN_SILENCE => props.turn_silence_threshold = value.get().unwrap(),
            PROP_MIN_TURN_SPEECH => props.min_turn_speech_duration = value.get().unwrap(),
            PROP_SPEECH_HYST => props.signal_speech_hysteresis = value.get().unwrap(),
            PROP_SILENCE_HYST => props.signal_silence_hysteresis = value.get().unwrap(),
            _ => {}
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let props = self.props.lock().unwrap();
        match pspec.name() {
            PROP_HF_REPO => props.hf_repo.to_value(),
            PROP_MODEL_PATH => props.model_path.to_value(),
            PROP_LANGUAGE => props.language.to_value(),
            PROP_USE_CPU => props.use_cpu.to_value(),
            PROP_TEMPERATURE => props.temperature.to_value(),
            PROP_INITIAL_PROMPT => props.initial_prompt.to_value(),
            PROP_DETECT_LANGUAGE => props.detect_language.to_value(),
            PROP_USE_UNIX_TS => props.use_unix_timestamps.to_value(),
            PROP_ENABLE_VAD => props.enable_vad.to_value(),
            PROP_VAD_THRESHOLD => props.vad_threshold.to_value(),
            PROP_NO_SPEECH_THRESHOLD => props.no_speech_threshold.to_value(),
            PROP_ENABLE_PAUSE => props.enable_pause_detection.to_value(),
            PROP_PAUSE_ENERGY => props.pause_energy_threshold.to_value(),
            PROP_PAUSE_MIN_DUR => props.pause_min_duration.to_value(),
            PROP_BOUNDARY_MIN_DUR => props.boundary_min_duration.to_value(),
            PROP_TURN_SILENCE => props.turn_silence_threshold.to_value(),
            PROP_MIN_TURN_SPEECH => props.min_turn_speech_duration.to_value(),
            PROP_SPEECH_HYST => props.signal_speech_hysteresis.to_value(),
            PROP_SILENCE_HYST => props.signal_silence_hysteresis.to_value(),
            _ => unreachable!(),
        }
    }
}

impl GstObjectImpl for KyutaiStt {}

impl ElementImpl for KyutaiStt {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Kyutai STT Transcriber",
                "Audio/Text/Filter",
                "Speech-to-text using Kyutai STT models (drop-in replacement for whispertranscribe)",
                "Kyutai",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let sink_pad = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &SINK_CAPS,
            )
            .unwrap();

            let src_pad = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &SRC_CAPS,
            )
            .unwrap();

            let ctrl_pad = gst::PadTemplate::new(
                "ctrl",
                gst::PadDirection::Sink,
                gst::PadPresence::Request,
                &CTRL_CAPS,
            )
            .unwrap();

            vec![sink_pad, src_pad, ctrl_pad]
        });
        PAD_TEMPLATES.as_ref()
    }

    fn change_state(
        &self,
        transition: gst::StateChange,
    ) -> Result<gst::StateChangeSuccess, gst::StateChangeError> {
        match transition {
            gst::StateChange::NullToReady => {
                if let Err(e) = self.load_model() {
                    let err_msg = format!("{e:#}");
                    self.obj()
                        .emit_by_name::<()>("model-load-failed", &[&err_msg]);
                    gst::error!(
                        gst::CAT_DEFAULT,
                        imp = self,
                        "Failed to load model: {err_msg}"
                    );
                    return Err(gst::StateChangeError);
                }
            }
            gst::StateChange::ReadyToNull => {
                *self.state.lock().unwrap() = None;
                self.obj().emit_by_name::<()>("model-unloaded", &[]);
            }
            _ => {}
        }
        self.parent_change_state(transition)
    }

    fn request_new_pad(
        &self,
        templ: &gst::PadTemplate,
        _name: Option<&str>,
        _caps: Option<&gst::Caps>,
    ) -> Option<gst::Pad> {
        if templ.name_template() == "ctrl" {
            let pad = gst::Pad::builder_from_template(templ)
                .chain_function(|pad, parent, buffer| {
                    KyutaiStt::catch_panic_pad_function(parent, || Err(gst::FlowError::Error), |this| {
                        this.handle_ctrl_buffer(pad, buffer)
                    })
                })
                .build();
            // The pad must be activated before it can be used.
            pad.set_active(true).ok()?;
            self.obj().add_pad(&pad).ok()?;
            Some(pad)
        } else {
            None
        }
    }

    fn release_pad(&self, pad: &gst::Pad) {
        pad.set_active(false).ok();
        self.obj().remove_pad(pad).ok();
    }
}

// ── BaseTransform impl ────────────────────────────────────────────────────

impl BaseTransformImpl for KyutaiStt {
    const MODE: gst_base::subclass::BaseTransformMode = gst_base::subclass::BaseTransformMode::Both;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_caps(
        &self,
        direction: gst::PadDirection,
        _caps: &gst::Caps,
        _filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        match direction {
            gst::PadDirection::Sink => Some(SRC_CAPS.clone()),
            gst::PadDirection::Src => Some(SINK_CAPS.clone()),
            _ => None,
        }
    }

    fn set_caps(&self, incaps: &gst::Caps, _outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let s = incaps.structure(0).unwrap();
        let rate: i32 = s.get("rate").unwrap();
        let channels: i32 = s.get("channels").unwrap();
        let format: String = s.get("format").unwrap();

        let mut state_guard = self.state.lock().unwrap();
        if let Some(state) = state_guard.as_mut() {
            state.input_rate = rate as u32;
            state.input_channels = channels as u32;

            // Set up resampler if needed.
            if rate as u32 != MODEL_SAMPLE_RATE {
                let resampler = rubato::FftFixedIn::<f32>::new(
                    rate as usize,
                    MODEL_SAMPLE_RATE as usize,
                    CHUNK_SAMPLES,
                    1,  // sub_chunks
                    1,  // mono (we downmix before resampling)
                )
                .map_err(|e| gst::loggable_error!(gst::CAT_DEFAULT, "Resampler init: {e}"))?;
                state.resampler = Some(resampler);
            } else {
                state.resampler = None;
            }

            gst::info!(
                gst::CAT_DEFAULT,
                imp = self,
                "Negotiated caps: format={format} rate={rate} channels={channels}"
            );
        }

        Ok(())
    }

    fn transform(
        &self,
        inbuf: &gst::Buffer,
        outbuf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let map = inbuf.map_readable().map_err(|_| gst::FlowError::Error)?;
        let data = map.as_slice();

        let mut state_guard = self.state.lock().unwrap();
        let state = state_guard.as_mut().ok_or(gst::FlowError::NotNegotiated)?;

        // Convert input bytes to f32 samples.
        let samples = self.bytes_to_f32(data, state);

        // Downmix to mono if stereo.
        let mono = if state.input_channels == 2 {
            samples
                .chunks_exact(2)
                .map(|pair| (pair[0] + pair[1]) * 0.5)
                .collect::<Vec<_>>()
        } else {
            samples
        };

        // Resample if needed.
        let resampled = if let Some(ref mut resampler) = state.resampler {
            use rubato::Resampler;
            let input = vec![mono.clone()];
            match resampler.process(&input, None) {
                Ok(output) => output.into_iter().next().unwrap_or_default(),
                Err(_) => mono,
            }
        } else {
            mono
        };

        // Append to PCM buffer and process in chunks.
        state.pcm_buffer.extend_from_slice(&resampled);

        let enable_pause = self.props.lock().unwrap().enable_pause_detection;
        let no_speech_thr = self.props.lock().unwrap().no_speech_threshold as f64;

        let mut all_words: Vec<crate::model::SttWord> = Vec::new();

        while state.pcm_buffer.len() >= CHUNK_SAMPLES {
            let chunk: Vec<f32> = state.pcm_buffer.drain(..CHUNK_SAMPLES).collect();

            // Pause/turn-taking detection on raw audio.
            if enable_pause {
                let pause_events = state.pause_detector.process(&chunk, MODEL_SAMPLE_RATE);
                self.emit_pause_events(&pause_events);
            }

            // STT inference.
            match state.model.process_chunk(&chunk) {
                Ok(events) => {
                    for event in events {
                        match event {
                            SttEvent::Word(word) => {
                                all_words.push(word);
                            }
                            SttEvent::Vad(vad) => {
                                // Use the 2s horizon (index 2) as the primary VAD signal.
                                let is_speech = if vad.no_speech_probs.len() > 2 {
                                    vad.no_speech_probs[2] < no_speech_thr
                                } else {
                                    true
                                };
                                let ts = (state.sample_position as i64 * 1_000_000_000)
                                    / MODEL_SAMPLE_RATE as i64;
                                self.obj().emit_by_name::<()>(
                                    "vad-speech-detected",
                                    &[&ts, &is_speech],
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    gst::warning!(
                        gst::CAT_DEFAULT,
                        imp = self,
                        "Inference error: {e:#}"
                    );
                }
            }

            state.sample_position += CHUNK_SAMPLES as u64;
        }

        // Build JSON output.
        let json = self.build_json_output(&all_words, state);
        let json_bytes = json.as_bytes();

        // Write to output buffer.
        let mut outmap = outbuf.map_writable().map_err(|_| gst::FlowError::Error)?;
        let out_slice = outmap.as_mut_slice();
        let copy_len = json_bytes.len().min(out_slice.len());
        out_slice[..copy_len].copy_from_slice(&json_bytes[..copy_len]);
        drop(outmap);

        outbuf.set_size(copy_len);

        // Emit segment-transcribed for each word.
        for word in &all_words {
            let s = gst::Structure::builder("segment")
                .field("text", &word.text)
                .field("start-time", word.start_time)
                .field("end-time", word.end_time)
                .field("confidence", word.confidence)
                .build();
            self.obj()
                .emit_by_name::<()>("segment-transcribed", &[&s]);
        }

        Ok(gst::FlowSuccess::Ok)
    }

    fn transform_size(
        &self,
        _direction: gst::PadDirection,
        _caps: &gst::Caps,
        _size: usize,
        _othercaps: &gst::Caps,
    ) -> Option<usize> {
        // JSON output: allocate 64 KB per transform call (matches whisper element).
        Some(65536)
    }

    fn sink_event(&self, event: gst::Event) -> bool {
        use gst::EventView;
        if let EventView::Eos(_) = event.view() {
            self.handle_eos();
        }
        self.parent_sink_event(event)
    }
}

// ── Private helpers ────────────────────────────────────────────────────────

impl KyutaiStt {
    fn load_model(&self) -> anyhow::Result<()> {
        let props = self.props.lock().unwrap().clone_for_init();

        let (model, info) = SttModel::load(
            &props.hf_repo,
            &props.model_path,
            props.enable_vad,
            props.use_cpu,
        )?;

        let pause_cfg = PauseDetectorConfig {
            energy_threshold: props.pause_energy_threshold,
            min_pause_ms: props.pause_min_duration as u64,
            min_boundary_ms: props.boundary_min_duration as u64,
            turn_silence_ms: props.turn_silence_threshold as u64,
            min_turn_speech_ms: props.min_turn_speech_duration as u64,
            speech_hysteresis_ms: props.signal_speech_hysteresis as u64,
            silence_hysteresis_ms: props.signal_silence_hysteresis as u64,
        };

        let model_info_struct = gst::Structure::builder("model-info")
            .field("type", "kyutai-stt")
            .field("sample-rate", info.sample_rate as i32)
            .field("gpu-enabled", info.gpu_enabled)
            .field("hf-repo", &info.hf_repo)
            .field("quantized", info.quantized)
            .build();

        let repo_name = info.hf_repo.clone();

        *self.state.lock().unwrap() = Some(State {
            model,
            info,
            pause_detector: PauseDetector::new(pause_cfg),
            pcm_buffer: Vec::with_capacity(CHUNK_SAMPLES * 4),
            segment_idx: 0,
            sample_position: 0,
            resampler: None,
            input_rate: MODEL_SAMPLE_RATE,
            input_channels: 1,
        });

        self.obj()
            .emit_by_name::<()>("model-loaded", &[&repo_name]);
        self.obj()
            .emit_by_name::<()>("model-info", &[&model_info_struct]);

        Ok(())
    }

    /// Convert raw bytes to f32 samples based on negotiated format.
    fn bytes_to_f32(&self, data: &[u8], _state: &State) -> Vec<f32> {
        // Detect format from the sink pad caps.
        let sink_pad = self.obj().static_pad("sink").unwrap();
        let caps = sink_pad.current_caps();
        let format = caps
            .as_ref()
            .and_then(|c| c.structure(0))
            .and_then(|s| s.get::<String>("format").ok())
            .unwrap_or_else(|| "F32LE".to_string());

        match format.as_str() {
            "S16LE" => data
                .chunks_exact(2)
                .map(|b| {
                    let sample = i16::from_le_bytes([b[0], b[1]]);
                    sample as f32 / 32768.0
                })
                .collect(),
            _ => {
                // F32LE
                data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
        }
    }

    /// Build whisper-compatible JSON output from collected words.
    fn build_json_output(
        &self,
        words: &[crate::model::SttWord],
        state: &mut State,
    ) -> String {
        if words.is_empty() {
            return r#"{"segments":[]}"#.to_string();
        }

        let segments: Vec<serde_json::Value> = words
            .iter()
            .map(|w| {
                state.segment_idx += 1;
                let tokens = vec![serde_json::json!({
                    "text": w.text,
                    "prob": w.confidence,
                    "start_ms": (w.start_time * 1000.0) as i64,
                    "end_ms": (w.end_time * 1000.0) as i64,
                    "start_sample": (w.start_time * MODEL_SAMPLE_RATE as f64) as i64,
                    "end_sample": (w.end_time * MODEL_SAMPLE_RATE as f64) as i64,
                })];
                serde_json::json!({
                    "text": w.text,
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "confidence": w.confidence,
                    "language": "auto",
                    "tokens": tokens,
                })
            })
            .collect();

        serde_json::json!({ "segments": segments }).to_string()
    }

    /// Emit GObject signals for pause/turn events.
    fn emit_pause_events(&self, events: &[PauseEvent]) {
        for event in events {
            match event {
                PauseEvent::SpeechStart { timestamp_ns } => {
                    self.obj()
                        .emit_by_name::<()>("speech-start", &[&(*timestamp_ns as i64)]);
                }
                PauseEvent::SpeechEnd {
                    timestamp_ns,
                    duration_ns,
                } => {
                    self.obj().emit_by_name::<()>(
                        "speech-end",
                        &[&(*timestamp_ns as i64), &(*duration_ns as i64)],
                    );
                }
                PauseEvent::PauseDetected {
                    timestamp_ns,
                    duration_ns,
                    energy,
                } => {
                    self.obj().emit_by_name::<()>(
                        "pause-detected",
                        &[&(*timestamp_ns as i64), &(*duration_ns as i64), energy],
                    );
                }
                PauseEvent::BoundaryDetected {
                    timestamp_ns,
                    confidence,
                } => {
                    self.obj().emit_by_name::<()>(
                        "boundary-detected",
                        &[&(*timestamp_ns as i64), confidence],
                    );
                }
                PauseEvent::TurnEnd {
                    timestamp_ns,
                    speech_duration_ns,
                    silence_ns,
                } => {
                    self.obj().emit_by_name::<()>(
                        "turn-end",
                        &[
                            &(*timestamp_ns as i64),
                            &(*speech_duration_ns as i64),
                            &(*silence_ns as i64),
                        ],
                    );
                }
                PauseEvent::SilenceDetected {
                    timestamp_ns,
                    duration_ns,
                } => {
                    self.obj().emit_by_name::<()>(
                        "silence-detected",
                        &[&(*timestamp_ns as i64), &(*duration_ns as i64)],
                    );
                }
            }
        }
    }

    /// Handle EOS: flush remaining audio through the model.
    fn handle_eos(&self) {
        let mut state_guard = self.state.lock().unwrap();
        if let Some(state) = state_guard.as_mut() {
            match state.model.flush() {
                Ok(events) => {
                    for event in events {
                        if let SttEvent::Word(word) = event {
                            let s = gst::Structure::builder("segment")
                                .field("text", &word.text)
                                .field("start-time", word.start_time)
                                .field("end-time", word.end_time)
                                .field("confidence", word.confidence)
                                .build();
                            self.obj()
                                .emit_by_name::<()>("segment-transcribed", &[&s]);
                        }
                    }
                }
                Err(e) => {
                    gst::warning!(
                        gst::CAT_DEFAULT,
                        imp = self,
                        "Flush error: {e:#}"
                    );
                }
            }
        }
    }

    /// Handle a buffer on the ctrl pad (runtime control commands).
    fn handle_ctrl_buffer(
        &self,
        _pad: &gst::Pad,
        buffer: gst::Buffer,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
        let json_str =
            std::str::from_utf8(map.as_slice()).map_err(|_| gst::FlowError::Error)?;

        match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(cmd) => {
                let command = cmd
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                match command {
                    "set_language" => {
                        if let Some(lang) = cmd.get("language").and_then(|v| v.as_str()) {
                            self.props.lock().unwrap().language = lang.to_string();
                            gst::info!(
                                gst::CAT_DEFAULT,
                                imp = self,
                                "Language set to: {lang}"
                            );
                        }
                    }
                    other => {
                        gst::debug!(
                            gst::CAT_DEFAULT,
                            imp = self,
                            "Unknown ctrl command: {other}"
                        );
                    }
                }
            }
            Err(e) => {
                gst::warning!(
                    gst::CAT_DEFAULT,
                    imp = self,
                    "Invalid ctrl JSON: {e}"
                );
            }
        }
        Ok(gst::FlowSuccess::Ok)
    }
}

/// Helper to extract property values for model init without holding the lock.
impl Properties {
    fn clone_for_init(&self) -> PropertiesSnapshot {
        PropertiesSnapshot {
            hf_repo: self.hf_repo.clone(),
            model_path: self.model_path.clone(),
            enable_vad: self.enable_vad,
            use_cpu: self.use_cpu,
            pause_energy_threshold: self.pause_energy_threshold,
            pause_min_duration: self.pause_min_duration,
            boundary_min_duration: self.boundary_min_duration,
            turn_silence_threshold: self.turn_silence_threshold,
            min_turn_speech_duration: self.min_turn_speech_duration,
            signal_speech_hysteresis: self.signal_speech_hysteresis,
            signal_silence_hysteresis: self.signal_silence_hysteresis,
        }
    }
}

struct PropertiesSnapshot {
    hf_repo: String,
    model_path: String,
    enable_vad: bool,
    use_cpu: bool,
    pause_energy_threshold: f64,
    pause_min_duration: u32,
    boundary_min_duration: u32,
    turn_silence_threshold: u32,
    min_turn_speech_duration: u32,
    signal_speech_hysteresis: u32,
    signal_silence_hysteresis: u32,
}
