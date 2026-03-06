;;; guix.scm --- Guix package definition for gst-kyutaistt
;;; This file can be used to test gst-kyutaistt locally with Guix:
;;;
;;;   guix shell -D -f guix.scm          # Development environment
;;;   guix build -f guix.scm             # Build the package
;;;
;;; After building or in a shell, set the plugin path:
;;;   export GST_PLUGIN_PATH=$GUIX_ENVIRONMENT/lib/gstreamer-1.0
;;;   gst-inspect-1.0 kyutaistt
(define-module (gst-kyutaistt)
  #:use-module (guix packages)
  #:use-module (guix gexp)
  #:use-module (guix git-download)
  #:use-module (guix utils)
  #:use-module (guix build-system cargo)
  #:use-module ((guix licenses) #:prefix license:)
  #:use-module (gnu packages)
  #:use-module (gnu packages cmake)
  #:use-module (gnu packages gstreamer)
  #:use-module (gnu packages pkg-config)
  #:use-module (gnu packages rust)
  #:use-module (gnu packages crates-io))

(define vcs-file?
  ;; Return true if the given file is under version control.
  (or (git-predicate (current-source-directory))
      (const #t)))                                ; not in a Git checkout

(define-public gst-kyutaistt
  (package
    (name "gst-kyutaistt")
    (version "0.1.0")
    (source
     (local-file "." "gst-kyutaistt-checkout"
                 #:recursive? #t
                 #:select? vcs-file?))
    (build-system cargo-build-system)
    (arguments
     (list
      #:cargo-inputs
      `(("rust-anyhow" ,rust-anyhow-1)
        ("rust-candle-core" ,rust-candle-core-0.9)
        ("rust-candle-nn" ,rust-candle-nn-0.9)
        ("rust-candle-transformers" ,rust-candle-transformers-0.9)
        ("rust-gstreamer" ,rust-gstreamer-0.23)
        ("rust-gstreamer-base" ,rust-gstreamer-base-0.23)
        ("rust-gstreamer-audio" ,rust-gstreamer-audio-0.23)
        ("rust-glib" ,rust-glib-0.20)
        ("rust-once-cell" ,rust-once-cell-1)
        ("rust-hf-hub" ,rust-hf-hub-0.4)
        ("rust-kaudio" ,rust-kaudio-0.2)
        ("rust-moshi" ,rust-moshi-0.6)
        ("rust-rubato" ,rust-rubato-0.16)
        ("rust-sentencepiece" ,rust-sentencepiece-0.11)
        ("rust-serde" ,rust-serde-1)
        ("rust-serde-json" ,rust-serde-json-1))
      #:cargo-build-flags ''("--release")
      #:install-source? #f
      #:phases
      #~(modify-phases %standard-phases
          (add-after 'install 'install-plugin
            (lambda* (#:key outputs #:allow-other-keys)
              (let* ((out (assoc-ref outputs "out"))
                     (gst-plugin-dir
                      (string-append out "/lib/gstreamer-1.0")))
                (mkdir-p gst-plugin-dir)
                (install-file "target/release/libgstkyutaistt.so"
                              gst-plugin-dir)))))))
    (native-inputs
     (list cmake                                  ; for sentencepiece-sys
           pkg-config))
    (inputs
     (list gstreamer
           gst-plugins-base))
    (home-page
     "https://huggingface.co/kyutai")
    (synopsis
     "GStreamer element for Kyutai STT speech-to-text")
    (description
     "gst-kyutaistt is a GStreamer element that performs streaming
speech-to-text using Kyutai STT models.  It is written entirely in Rust
with no Python dependency.  Inference runs through Candle, audio
tokenisation uses the Mimi codec, and text decoding uses SentencePiece.

Features:
@itemize
@item Native streaming transcription in 80 ms chunks
@item Dual VAD: built-in neural (semantic) VAD head plus energy-based
pause detection
@item Turn-end detection for voice agent integration
@item Support for 1B (English/French) and 2.6B (English) models
@item GPU acceleration via CUDA, cuDNN, or Metal
@item Drop-in replacement for whispertranscribe (compatible pad caps,
JSON output format, and signal interface)
@end itemize")
    (license (list license:expat
                   license:asl2.0))))

;; Return the package for use with 'guix shell -f guix.scm'
gst-kyutaistt
