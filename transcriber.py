"""Speech-to-text: Whisper (default), Parakeet v3, or Parakeet double-checked by Whisper.

Measured on a real recorded session (short commands and gags, 13 clips):
Whisper forced to Spanish got 11-12 right in ~205 ms each (max 234 ms);
Parakeet got 3 right, so the hybrid mostly ended up running both.
Parakeet's strength is long sentences from several people at once.

Parakeet (NVIDIA's parakeet-tdt-0.6b-v3 through sherpa-onnx, int8 on the CPU)
does the work: ~130 ms per sentence, no GPU memory, and it decodes several
people's sentences in one batch. It picks the language by itself, though, and
on short or noisy clips it sometimes drifts ("Sí" -> "See?", Spanish heard as
Finnish). HybridTranscriber re-does those few with Whisper (large-v3-turbo on
the GPU) forced to the expected language: low Parakeet confidence, letters
that language doesn't use, or none of its common words at all. Both models
are downloaded to the Hugging Face cache on first use.

TranscriptionBatcher sits in front of either engine: whatever is waiting when
the engine frees up is transcribed together. Each clip can say which language
to expect (the server's); Whisper is then forced to it.
"""
import concurrent.futures
import ctypes
import glob
import io
import logging
import os
import queue
import re
import site
import threading
import unicodedata
import wave

import numpy as np
import soxr

log = logging.getLogger(__name__)

DEFAULT_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
PARAKEET_REPO = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
# Full precision, for the GPU (int8 barely runs faster there).
PARAKEET_GPU_REPO = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SECONDARY_LANGUAGE_MIN_PROB = 0.8

# Discord delivers 48 kHz stereo 16-bit PCM.
DISCORD_RATE = 48000
DISCORD_CHANNELS = 2

# Speech-to-text stock inventions on breathing, music and silence (Whisper's
# training data was full of video subtitles). Compared after normalize(),
# against the whole utterance.
# Nobody says these in a call: always dropped.
HALLUCINATIONS = {
    "thanks for watching", "thank you for watching", "please subscribe", "like and subscribe",
    "gracias por ver", "gracias por ver el video", "gracias por ver el video hasta el final",
    "suscribete", "suscribete al canal", "no olvides suscribirte",
    "subtitulos realizados por la comunidad de amara org", "subtitulos por la comunidad de amara org",
    "subtitulado por la comunidad de amara org", "amara org",
}
# Real words that Whisper also invents out of near-silence ("Gracias." on a
# cough): kept only when they were clearly heard (see judge()).
# (Not "chao", "bye" & co.: those can be a real one-word follow-up to the
# wake word, and are short enough to fail the checks.)
DOUBTFUL = {
    "gracias", "muchas gracias", "gracias a todos",
    "thank you", "thanks", "thank you so much", "thank you very much", "you",
    "mm", "mmm", "hmm", "eh", "ah", "oh", "uh",
}
# Text in any segment that only ever comes from subtitle credits.
CREDIT_MARKERS = ("amara org", "subtitulos realizados", "subtitulado por", "suscribete")
# Below this much actual speech (voice activity detection), a clip is noise:
# not even sent to Whisper, which would make something up.
MIN_SPEECH_S = 0.3
# A doubtful phrase counts as heard only with at least this much speech, and
# when Whisper is this sure there was speech / of the words.
DOUBTFUL_MIN_SPEECH_S = 0.4
DOUBTFUL_MAX_NO_SPEECH = 0.15
DOUBTFUL_MIN_LOGPROB = -0.4
# A segment Whisper thinks is silence and isn't sure about, or that repeats
# itself (compression ratio), is dropped.
SEGMENT_MAX_NO_SPEECH = 0.6
SEGMENT_MIN_LOGPROB = -0.7
SEGMENT_MAX_COMPRESSION = 2.4


class Heard(tuple):
    """A transcript: unpacks as (text, language) like before, and carries how
    sure the engine was. confidence: Whisper's mean log-probability of the
    words (0 = certain, below about -0.7 = mumbled or guessed); None when the
    engine doesn't measure it the same way (Parakeet). dropped: why a
    transcript was thrown away (then text is ""), e.g. "hallucination"."""

    def __new__(cls, text: str, language: str, *, confidence: float | None = None,
                no_speech: float | None = None, dropped: str | None = None, heard: str = ""):
        self = super().__new__(cls, (text, language))
        self.confidence, self.no_speech, self.dropped = confidence, no_speech, dropped
        self.heard = heard or text  # what the engine produced, even if dropped
        return self

    @property
    def text(self) -> str:
        return self[0]

    @property
    def language(self) -> str:
        return self[1]


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation: "¡Desconéctate!" -> "desconectate"."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


# Parakeet's mean token log-probability below this means "probably wrong":
# on a noisy test set it flagged 25 of 33 wrong transcripts and 4 of 121
# right ones.
MIN_CONFIDENCE = -0.15
# Per language: the letters beyond a-z it uses (any other one is a sure sign
# of a language drift), and very common words (a longer transcript with none
# of them isn't in that language). Languages missing here are only checked
# by confidence.
LANGUAGE_LETTERS = {
    "es": set("áéíóúüñ"),
    "en": set(),
}
COMMON_WORDS = {
    "es": frozenset("""
el la los las lo le les un una unos unas de del al a en y e o u que qué no ni si sí es son era fue
ser estar esta este esto esa ese eso está estoy estás están hay ya me te se mi mis tu tus su sus
nos yo tú él ella ellos usted vos con por para sin como cómo cuando donde dónde pero porque más muy
bien mal todo nada algo vamos voy va vas dale vale pues oye mira bueno ahí aquí allá eh ah ajá
tengo tiene tienes hace hacer ver dice dijo sé sabe quiero puedo
""".split()),
    "en": frozenset("""
the a an to of and in is it you that i he she we they was for on are with be at this have not but
what so do my no yes yeah ok okay just like your can go get me oh hey
""".split()),
}


def looks_wrong(text: str, confidence: float, languages=("es", "en")) -> bool:
    """Should this Parakeet transcript be double-checked by Whisper? It
    should be in one of `languages`."""
    if confidence < MIN_CONFIDENCE:
        return True
    known = [lang for lang in languages if lang in LANGUAGE_LETTERS]
    if not known:
        return False
    allowed = set().union(*(LANGUAGE_LETTERS[lang] for lang in known))
    if any(c.isalpha() and not c.isascii() and c not in allowed for c in text.lower()):
        return True
    common = frozenset().union(*(COMMON_WORDS[lang] for lang in known))
    words = re.findall(r"[^\W\d_]+", text.lower())
    return len(words) >= 3 and not any(w in common for w in words)


def clean(text: str) -> str:
    text = text.strip()
    if normalize(text) in HALLUCINATIONS:
        log.debug("Dropping likely hallucination: %r", text)
        return ""
    return text


def judge(segments: list[tuple[str, float, float, float]], speech_s: float) -> tuple[str, float | None, float | None, str | None]:
    """Decide what of Whisper's output to believe.
    segments: (text, avg_logprob, no_speech_prob, compression_ratio) each.
    speech_s: seconds of actual speech in the clip (voice activity detection).
    Returns (text, confidence, no_speech, why_dropped)."""
    credits = [s for s in segments if any(marker in normalize(s[0]) for marker in CREDIT_MARKERS)]
    kept = [s for s in segments
            if s[0].strip() and s not in credits
            and not (s[2] > SEGMENT_MAX_NO_SPEECH and s[1] < SEGMENT_MIN_LOGPROB)
            and s[3] <= SEGMENT_MAX_COMPRESSION]
    text = " ".join(s[0].strip() for s in kept).strip()
    if not text:
        return "", None, None, ("hallucination" if credits else "noise") if segments else None
    # Weighted by length: one long clear sentence outweighs a mumbled "eh".
    weights = [max(len(s[0].strip()), 1) for s in kept]
    confidence = sum(s[1] * w for s, w in zip(kept, weights)) / sum(weights)
    no_speech = max(s[2] for s in kept)
    norm = normalize(text)
    if norm in HALLUCINATIONS:
        return "", confidence, no_speech, "hallucination"
    if norm in DOUBTFUL and (speech_s < DOUBTFUL_MIN_SPEECH_S or no_speech > DOUBTFUL_MAX_NO_SPEECH
                             or confidence < DOUBTFUL_MIN_LOGPROB):
        return "", confidence, no_speech, "hallucination"
    return text, confidence, no_speech, None


_vad_options = None


def speech_seconds(audio: np.ndarray) -> float:
    """Seconds of speech in a 16 kHz clip, by Silero VAD (CPU, a few ms)."""
    global _vad_options
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    if _vad_options is None:
        _vad_options = VadOptions(min_speech_duration_ms=150, min_silence_duration_ms=300, speech_pad_ms=30)
    return sum(s["end"] - s["start"] for s in get_speech_timestamps(audio, _vad_options)) / 16000


def pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(DISCORD_CHANNELS)
        w.setsampwidth(2)
        w.setframerate(DISCORD_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def pcm_to_16k(pcm: bytes) -> np.ndarray:
    """Discord PCM -> 16 kHz mono float32, what speech models take."""
    stereo = np.frombuffer(pcm, dtype=np.int16).reshape(-1, DISCORD_CHANNELS)
    mono = stereo.astype(np.float32).mean(axis=1) / 32768
    return soxr.resample(mono, DISCORD_RATE, 16000).astype(np.float32)


def _gpu_model_dir() -> str:
    """The full-precision model, hard-linked out of the Hugging Face cache:
    onnxruntime refuses an encoder whose weights file (encoder.weights) sits
    in another directory, which is how the cache stores it."""
    from huggingface_hub import snapshot_download

    cached = snapshot_download(PARAKEET_GPU_REPO, allow_patterns=["*.onnx", "*.weights", "tokens.txt"])
    target = os.path.join(MODELS_DIR, "parakeet-tdt-0.6b-v3")
    os.makedirs(target, exist_ok=True)
    for name in ("encoder.onnx", "encoder.weights", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        link = os.path.join(target, name)
        if not os.path.exists(link):
            os.link(os.path.realpath(os.path.join(cached, name)), link)
    return target


class ParakeetTranscriber:
    """Parakeet v3. Spanish, English and ~23 other European languages.

    cpu: int8, ~135 ms a sentence, no GPU memory.
    gpu: full precision, ~46 ms a sentence, 3.4 GB of GPU memory.
    """

    def __init__(self, device: str = "cpu", num_threads: int = 4):
        self.device = device
        # 4 measured fastest on a 20-thread i7-12700H (8 was slower).
        self.num_threads = num_threads
        self.recognizer = None

    def load(self) -> None:
        import sherpa_onnx
        from huggingface_hub import snapshot_download

        if self.device == "gpu":
            _preload_cuda_libs()
            model_dir, suffix, provider = _gpu_model_dir(), "", "cuda"
        else:
            model_dir, suffix, provider = snapshot_download(PARAKEET_REPO), ".int8", "cpu"
        path = lambda name: os.path.join(model_dir, name)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=path(f"encoder{suffix}.onnx"),
            decoder=path(f"decoder{suffix}.onnx"),
            joiner=path(f"joiner{suffix}.onnx"),
            tokens=path("tokens.txt"),
            num_threads=self.num_threads,
            provider=provider,
            model_type="nemo_transducer",
            decoding_method="greedy_search",
        )
        self.transcribe_audio(np.zeros(16000, dtype=np.float32))  # warm-up
        log.info("Loaded Parakeet v3 on the %s", self.device.upper())

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """16 kHz mono float32 -> text."""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def decode(self, audios: list[np.ndarray]) -> list[tuple[str, float]]:
        """16 kHz clips -> [(text, confidence)], decoded as one batch.
        Confidence is the mean token log-probability (0 = certain)."""
        streams = []
        for audio in audios:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, audio)
            streams.append(stream)
        self.recognizer.decode_streams(streams)
        return [
            (s.result.text.strip(), float(np.mean(s.result.ys_log_probs)) if s.result.ys_log_probs else 0.0)
            for s in streams
        ]

    def transcribe_many(self, pcms: list[bytes], languages=None) -> list[tuple[str, str]]:
        """Discord PCM clips -> [(text, language)], decoded as one batch.
        Parakeet picks the language by itself: `languages` is ignored."""
        # Parakeet doesn't report the language it heard. Its confidence is on
        # another scale than Whisper's, so it isn't passed on.
        results = []
        for text, _ in self.decode([pcm_to_16k(p) for p in pcms]):
            kept = clean(text)
            results.append(Heard(kept, "auto", dropped="hallucination" if text and not kept else None, heard=text))
        return results


def _preload_cuda_libs() -> None:
    """CTranslate2 (Whisper) and onnxruntime (Parakeet on the GPU) dlopen the
    CUDA libraries by soname, but pip installs them under
    site-packages/nvidia/*/lib, which isn't on the loader path. Load them into
    the process first so their dlopen finds them already resident.
    """
    patterns = (
        "nvidia/cuda_runtime/lib/libcudart.so.*",
        "nvidia/cublas/lib/libcublasLt.so.*",
        "nvidia/cublas/lib/libcublas.so.*",
        "nvidia/cudnn/lib/libcudnn*.so.*",
        "nvidia/cufft/lib/libcufft.so.*",
        "nvidia/curand/lib/libcurand.so.*",
    )
    for sp in site.getsitepackages():
        for pattern in patterns:
            for lib in sorted(glob.glob(os.path.join(sp, pattern))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError as e:
                    log.debug("Could not preload %s: %s", lib, e)


class WhisperTranscriber:
    """faster-whisper large-v3-turbo on the GPU (~2.3 GB), one clip at a time,
    ~205 ms for a short clip when forced to one language."""

    def __init__(self, model_name: str = DEFAULT_MODEL, languages: list[str] | None = None,
                 hotwords: str | None = None, cpu_only: bool = False):
        self.model_name = model_name
        self.cpu_only = cpu_only
        # Languages people actually speak; detection only picks among these.
        # Empty = let Whisper choose from all ~100.
        self.languages = languages or []
        self.hotwords = hotwords
        self.beam_size = 5
        self.model = None
        self.device = self.compute_type = None
        # Called when the GPU runs out of memory, before one retry (the bot
        # points it at the voice model's cache release).
        self.on_oom = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        _preload_cuda_libs()
        # Fastest GPU type first, CPU last.
        options = (("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8"))
        for device, compute_type in options[2:] if self.cpu_only else options:
            try:
                self.model = WhisperModel(self.model_name, device=device, compute_type=compute_type)
            except Exception as e:
                log.warning("Loading %s on %s/%s failed: %s", self.model_name, device, compute_type, e)
                continue
            self.device, self.compute_type = device, compute_type
            log.info("Loaded %s on %s (%s)", self.model_name, device, compute_type)
            return
        raise RuntimeError(f"Could not load {self.model_name} on any device")

    def _pick_language(self, audio) -> str | None:
        if len(self.languages) <= 1:
            return self.languages[0] if self.languages else None
        # Whisper guesses from ~1 s of audio, so on short clips it happily
        # answers Portuguese or Finnish for Spanish. Restrict it to our set.
        try:
            _, _, probs = self.model.detect_language(audio, vad_filter=True)
        except Exception as e:
            log.debug("Language detection failed, using %s: %s", self.languages[0], e)
            return self.languages[0]
        probs = dict(probs)
        # The first language is the default. Another one has to be clear-cut:
        # real English scores ~1.0, while Spanish with an English word in it
        # ("Sí, mi Lord") can still score 0.6 English, and a wrong language
        # makes Whisper translate instead of transcribe.
        primary, *others = self.languages
        best = max(others, key=lambda lang: probs.get(lang, 0.0))
        return best if probs.get(best, 0.0) >= SECONDARY_LANGUAGE_MIN_PROB else primary

    def _transcribe(self, audio: np.ndarray, language: str | None) -> Heard:
        # Almost no speech (a cough, a click, a breath): Whisper would invent
        # something ("Gracias."), so it isn't asked at all.
        speech_s = speech_seconds(audio)
        if speech_s < MIN_SPEECH_S:
            return Heard("", language or "", dropped="noise")
        segments, info = self.model.transcribe(
            audio,
            language=language,
            hotwords=self.hotwords,
            beam_size=self.beam_size,
            # No retries at higher temperatures and no timestamps: on real clips
            # 437 ms -> 205 ms on average (977 -> 234 worst), same text.
            temperature=0.0,
            without_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            # Each utterance stands alone; carrying context over makes Whisper
            # repeat itself across speakers.
            condition_on_previous_text=False,
        )
        segments = [(s.text, s.avg_logprob, s.no_speech_prob, s.compression_ratio) for s in segments]
        text, confidence, no_speech, dropped = judge(segments, speech_s)
        if dropped == "hallucination":
            log.info("Ignoring %r: likely made up (speech %.2fs, confidence %.2f, no-speech %.2f)",
                     " ".join(s[0].strip() for s in segments), speech_s, confidence, no_speech)
        return Heard(text, info.language, confidence=confidence, no_speech=no_speech, dropped=dropped,
                     heard=" ".join(s[0].strip() for s in segments).strip())

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """For references (a voice sample): no hallucination checks, just the words."""
        segments, _ = self.model.transcribe(audio, language=self.languages[0] if self.languages else None,
                                            beam_size=self.beam_size, temperature=0.0, without_timestamps=True,
                                            vad_filter=True, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe_one(self, audio: np.ndarray, language: str | None = None) -> Heard:
        """16 kHz clip -> (text, language). language: force it (e.g. the
        server's); None = pick among self.languages."""
        try:
            return self._transcribe(audio, language or self._pick_language(audio))
        except RuntimeError as e:
            if "out of memory" not in str(e) or self.on_oom is None:
                raise
            log.warning("GPU out of memory while transcribing; freeing memory and retrying")
            self.on_oom()
            return self._transcribe(audio, language or self._pick_language(audio))

    def transcribe_many(self, pcms: list[bytes], languages=None) -> list[tuple[str, str]]:
        """languages: one per clip (None = this engine's default)."""
        languages = languages or [None] * len(pcms)
        return [self.transcribe_one(pcm_to_16k(pcm), lang) for pcm, lang in zip(pcms, languages)]


class HybridTranscriber:
    """Parakeet for everything; Whisper, forced to the expected language (the
    clip's, else Whisper's main one), re-does the transcripts that look wrong
    (see looks_wrong)."""

    def __init__(self, parakeet: ParakeetTranscriber, whisper: WhisperTranscriber):
        self.parakeet = parakeet
        self.whisper = whisper
        self.rechecked = self.total = 0

    @property
    def on_oom(self):
        return self.whisper.on_oom

    @on_oom.setter
    def on_oom(self, callback) -> None:
        self.whisper.on_oom = callback

    def load(self) -> None:
        self.parakeet.load()
        self.whisper.load()

    def transcribe_audio(self, audio: np.ndarray) -> str:
        return self.parakeet.transcribe_audio(audio)

    def transcribe_many(self, pcms: list[bytes], languages=None) -> list[tuple[str, str]]:
        audios = [pcm_to_16k(pcm) for pcm in pcms]
        languages = languages or [None] * len(pcms)
        results = []
        for audio, expected, (text, confidence) in zip(audios, languages, self.parakeet.decode(audios)):
            self.total += 1
            # An empty result gets re-checked too: the listener only sends
            # clips with speech in them, and Parakeet drops fast speech.
            if text and not looks_wrong(text, confidence, [expected] if expected else self.whisper.languages):
                kept = clean(text)
                results.append(Heard(kept, "auto", dropped="hallucination" if not kept else None, heard=text))
                continue
            self.rechecked += 1
            fixed = self.whisper.transcribe_one(audio, expected)
            # Whisper hearing nothing in a doubtful clip means it was noise.
            log.info("Re-checked %r (confidence %.2f) -> %r [%d/%d re-checked so far]",
                     text, confidence, fixed.text, self.rechecked, self.total)
            results.append(fixed)
        return results


class TranscriptionBatcher:
    """One worker thread in front of an engine. A clip submitted while the
    engine is idle starts right away; clips that arrive while it's busy are
    all transcribed together as soon as it frees up."""

    def __init__(self, engine, max_batch: int = 8):
        self.engine = engine
        self.max_batch = max_batch
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="stt-batcher", daemon=True).start()

    def submit(self, pcm: bytes, language: str | None = None) -> concurrent.futures.Future:
        """Transcribe one Discord PCM clip -> future of (text, language).
        language: what to expect (e.g. "es"); None = the engine's default."""
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((pcm, language, future))
        return future

    def _run(self) -> None:
        while True:
            batch = [self._queue.get()]
            while len(batch) < self.max_batch:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            try:
                # Mixed languages batch fine: each clip carries its own.
                results = self.engine.transcribe_many([pcm for pcm, _, _ in batch], [lang for _, lang, _ in batch])
            except Exception as e:
                for _, _, future in batch:
                    future.set_exception(e)
                continue
            if len(batch) > 1:
                log.debug("Transcribed %d utterances in one batch", len(batch))
            for (_, _, future), result in zip(batch, results):
                future.set_result(result)
