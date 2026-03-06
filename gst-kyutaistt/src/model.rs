// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! Wrapper around the Kyutai STT model, adapted from stt-rs/src/main.rs.

use anyhow::Result;
use candle::{Device, Tensor};

/// Decoded word with timing information.
#[derive(Debug, Clone)]
pub struct SttWord {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
    pub confidence: f64,
}

/// VAD probability update from the model's built-in VAD head.
#[derive(Debug, Clone)]
pub struct VadUpdate {
    /// Probability of no-voice-activity at various horizons.
    /// Index 0 = ~0.5s, 1 = ~1s, 2 = ~2s, 3 = ~3s.
    pub no_speech_probs: Vec<f64>,
}

/// Events emitted by the model during inference.
#[derive(Debug, Clone)]
pub enum SttEvent {
    Word(SttWord),
    Vad(VadUpdate),
}

#[derive(Debug, serde::Deserialize)]
struct SttConfig {
    audio_silence_prefix_seconds: f64,
    audio_delay_seconds: f64,
}

#[derive(Debug, serde::Deserialize)]
struct Config {
    mimi_name: String,
    tokenizer_name: String,
    card: usize,
    text_card: usize,
    dim: usize,
    n_q: usize,
    context: usize,
    max_period: f64,
    num_heads: usize,
    num_layers: usize,
    causal: bool,
    stt_config: SttConfig,
}

impl Config {
    fn model_config(&self, vad: bool) -> moshi::lm::Config {
        let lm_cfg = moshi::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dim_feedforward: self.dim * 4,
            causal: self.causal,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: self.context,
            max_period: self.max_period as usize,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: moshi::NormType::RmsNorm,
            positional_embedding: moshi::transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096 * 4,
            shared_cross_attn: false,
        };
        let extra_heads = if vad {
            Some(moshi::lm::ExtraHeadsConfig {
                num_heads: 4,
                dim: 6,
            })
        } else {
            None
        };
        moshi::lm::Config {
            transformer: lm_cfg,
            depformer: None,
            audio_vocab_size: self.card + 1,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_codebooks: self.n_q,
            conditioners: Default::default(),
            extra_heads,
        }
    }
}

pub struct SttModel {
    state: moshi::asr::State,
    text_tokenizer: sentencepiece::SentencePieceProcessor,
    config: Config,
    dev: Device,
    enable_vad: bool,
    /// Accumulated silence prefix samples remaining to prepend.
    silence_prefix_remaining: usize,
    /// Pending word that hasn't received an end time yet.
    pending_word: Option<(String, f64)>,
}

impl SttModel {
    /// Select the best available device.
    pub fn device(cpu: bool) -> Result<Device> {
        if cpu {
            Ok(Device::Cpu)
        } else if candle::utils::cuda_is_available() {
            Ok(Device::new_cuda(0)?)
        } else if candle::utils::metal_is_available() {
            Ok(Device::new_metal(0)?)
        } else {
            Ok(Device::Cpu)
        }
    }

    /// Load model from a HuggingFace repository.
    pub fn load(
        hf_repo: &str,
        model_path: &str,
        enable_vad: bool,
        cpu: bool,
    ) -> Result<(Self, ModelInfo)> {
        let dev = Self::device(cpu)?;
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(hf_repo.to_string());
        let config_file = repo.get("config.json")?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
        let tokenizer_file = repo.get(&config.tokenizer_name)?;
        let model_file = repo.get(model_path)?;
        let mimi_file = repo.get(&config.mimi_name)?;
        let is_quantized = model_file
            .to_str()
            .unwrap_or_default()
            .ends_with(".gguf");

        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&tokenizer_file)?;

        let lm = if is_quantized {
            let vb_lm =
                candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&model_file, &dev)?;
            moshi::lm::LmModel::new(
                &config.model_config(enable_vad),
                moshi::nn::MaybeQuantizedVarBuilder::Quantized(vb_lm),
            )?
        } else {
            let dtype = dev.bf16_default_to_f32();
            let vb_lm = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, &dev)?
            };
            moshi::lm::LmModel::new(
                &config.model_config(enable_vad),
                moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
            )?
        };

        let audio_tokenizer =
            moshi::mimi::load(mimi_file.to_str().unwrap(), Some(32), &dev)?;
        let asr_delay_in_tokens = (config.stt_config.audio_delay_seconds * 12.5) as usize;
        let state = moshi::asr::State::new(1, asr_delay_in_tokens, 0., audio_tokenizer, lm)?;

        let silence_prefix_remaining =
            (config.stt_config.audio_silence_prefix_seconds * 24000.0) as usize;

        let info = ModelInfo {
            sample_rate: 24000,
            gpu_enabled: !cpu && (candle::utils::cuda_is_available() || candle::utils::metal_is_available()),
            hf_repo: hf_repo.to_string(),
            quantized: is_quantized,
        };

        Ok((
            Self {
                state,
                text_tokenizer,
                config,
                dev,
                enable_vad,
                silence_prefix_remaining,
                pending_word: None,
            },
            info,
        ))
    }

    /// Process a chunk of PCM audio (f32, 24kHz, mono) and return events.
    /// Chunks should be 1920 samples (80ms) for optimal processing.
    pub fn process_chunk(&mut self, pcm: &[f32]) -> Result<Vec<SttEvent>> {
        let mut events = Vec::new();
        let mut input = pcm.to_vec();

        // Prepend silence prefix if still remaining.
        if self.silence_prefix_remaining > 0 {
            let prepend = self.silence_prefix_remaining.min(input.len());
            let mut silence = vec![0.0f32; prepend];
            silence.append(&mut input);
            input = silence;
            self.silence_prefix_remaining -= prepend;
        }

        // Process in 1920-sample sub-chunks (the model's native step size).
        for sub_chunk in input.chunks(1920) {
            let pcm_tensor =
                Tensor::new(sub_chunk, &self.dev)?.reshape((1, 1, sub_chunk.len()))?;
            let asr_msgs =
                self.state
                    .step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

            for msg in asr_msgs.iter() {
                match msg {
                    moshi::asr::AsrMsg::Step { prs, .. } => {
                        if self.enable_vad {
                            let no_speech_probs: Vec<f64> =
                                prs.iter().map(|p| p[0] as f64).collect();
                            events.push(SttEvent::Vad(VadUpdate { no_speech_probs }));
                        }
                    }
                    moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                        if let Some((word, start_time)) = self.pending_word.take() {
                            events.push(SttEvent::Word(SttWord {
                                text: word,
                                start_time,
                                end_time: *stop_time,
                                confidence: 1.0,
                            }));
                        }
                    }
                    moshi::asr::AsrMsg::Word {
                        tokens, start_time, ..
                    } => {
                        // Flush any previous pending word with this word's start as its end.
                        if let Some((prev_word, prev_start)) = self.pending_word.take() {
                            events.push(SttEvent::Word(SttWord {
                                text: prev_word,
                                start_time: prev_start,
                                end_time: *start_time,
                                confidence: 1.0,
                            }));
                        }
                        let word = self
                            .text_tokenizer
                            .decode_piece_ids(tokens)
                            .unwrap_or_default();
                        self.pending_word = Some((word, *start_time));
                    }
                }
            }
        }

        Ok(events)
    }

    /// Flush remaining audio through the model by feeding silence.
    /// Call this at EOS.
    pub fn flush(&mut self) -> Result<Vec<SttEvent>> {
        let suffix_samples =
            (self.config.stt_config.audio_delay_seconds * 24000.0) as usize + 24000;
        let silence = vec![0.0f32; suffix_samples];
        let mut events = self.process_chunk(&silence)?;
        // Flush any remaining pending word.
        if let Some((word, start_time)) = self.pending_word.take() {
            events.push(SttEvent::Word(SttWord {
                text: word,
                start_time,
                end_time: start_time + 0.08,
                confidence: 1.0,
            }));
        }
        Ok(events)
    }
}

/// Information about the loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub sample_rate: u32,
    pub gpu_enabled: bool,
    pub hf_repo: String,
    pub quantized: bool,
}
