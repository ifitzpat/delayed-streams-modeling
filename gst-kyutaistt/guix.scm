;;; guix.scm --- Guix package definition for gst-kyutaistt
;;;
;;; Development environment (recommended):
;;;
;;;   guix shell -D -f guix.scm
;;;   cargo build --release
;;;   export GST_PLUGIN_PATH=$PWD/target/release
;;;   gst-inspect-1.0 kyutaistt
;;;
;;; Full sandboxed build (requires vendored dependencies):
;;;
;;;   cargo vendor                          # run once, outside Guix
;;;   guix build -f guix.scm               # builds inside sandbox
;;;
;;; The sandboxed build requires a vendor/ directory because Guix builds
;;; run without network access, and none of the 17 direct Rust crate
;;; dependencies are packaged in Guix.  Run `cargo vendor` to populate
;;; vendor/ before `guix build`.  The vendor/ directory is included in
;;; the source via the vcs-file? predicate (commit it, or remove the
;;; predicate to include untracked files).
(define-module (gst-kyutaistt)
  #:use-module (guix packages)
  #:use-module (guix gexp)
  #:use-module (guix git-download)
  #:use-module (guix utils)
  #:use-module (guix build-system gnu)
  #:use-module ((guix licenses) #:prefix license:)
  #:use-module (gnu packages)
  #:use-module (gnu packages cmake)
  #:use-module (gnu packages gstreamer)
  #:use-module (gnu packages pkg-config)
  #:use-module (gnu packages rust)
  #:use-module (gnu packages tls))

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
    (build-system gnu-build-system)
    (arguments
     (list
      #:phases
      #~(modify-phases %standard-phases
          (delete 'configure)
          (replace 'build
            (lambda* (#:key inputs #:allow-other-keys)
              ;; Point Cargo at vendored dependencies (no network in sandbox)
              (setenv "CARGO_HOME" (string-append (getcwd) "/.cargo"))
              (mkdir-p ".cargo")
              (when (file-exists? "vendor")
                (call-with-output-file ".cargo/config.toml"
                  (lambda (port)
                    (format port
                            "[source.crates-io]~%~
                             replace-with = \"vendored-sources\"~%~
                             ~%~
                             [source.vendored-sources]~%~
                             directory = \"vendor\"~%"))))
              (invoke "cargo" "build" "--release")))
          (delete 'check)
          (replace 'install
            (lambda* (#:key outputs #:allow-other-keys)
              (let ((gst-plugin-dir
                     (string-append (assoc-ref outputs "out")
                                    "/lib/gstreamer-1.0")))
                (mkdir-p gst-plugin-dir)
                (install-file "target/release/libgstkyutaistt.so"
                              gst-plugin-dir)))))))
    (native-inputs
     (list cmake                                  ; for sentencepiece-sys
           pkg-config
           rust
           `(,rust "cargo")))
    (inputs
     (list gstreamer
           gst-plugins-base
           openssl))                              ; for hf-hub HTTPS downloads
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
