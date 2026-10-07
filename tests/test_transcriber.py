"""Speech-to-text plumbing with fake engines: no models are loaded."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import transcriber as stt
from transcriber import HybridTranscriber, TranscriptionBatcher, WhisperTranscriber, looks_wrong

PCM = np.zeros(48000 * 2, dtype=np.int16).tobytes()  # 1 s of Discord silence


@pytest.fixture(autouse=True)
def speech_everywhere(monkeypatch):
    """The fake clips are silence; pretend voice activity detection found
    2 s of speech unless a test says otherwise."""
    monkeypatch.setattr(stt, "speech_seconds", lambda audio: 2.0)


def test_looks_wrong_per_language():
    assert not looks_wrong("hola qué tal estás", 0.0, ["es"])
    assert not looks_wrong("hello how are you", 0.0, ["en"])
    assert looks_wrong("hyvää huomenta kaikille", 0.0, ["es"])       # letters Spanish doesn't use
    assert looks_wrong("qué tal amigos míos", 0.0, ["en"])            # accents English doesn't use
    assert not looks_wrong("qué tal amigos míos", 0.0, ["es", "en"])
    assert looks_wrong("hola qué tal", -0.5, ["es"])                  # low confidence
    assert looks_wrong("gato perro casa", 0.0, ["en"])                # no common English word
    assert not looks_wrong("ok", 0.0, ["en"])                         # too short to judge
    assert not looks_wrong("bonjour à tous mes amis", 0.0, ["fr"])    # unknown language: confidence only


class FakeWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None, **kwargs):
        self.calls.append(language)
        return [SimpleNamespace(text=f"heard in {language}", no_speech_prob=0.0, avg_logprob=0.0,
                                compression_ratio=1.0)], SimpleNamespace(language=language)

    def detect_language(self, audio, **kwargs):
        return None, None, [("es", 0.1), ("en", 0.9)]


def whisper(languages):
    w = WhisperTranscriber(languages=languages)
    w.model = FakeWhisperModel()
    return w


def test_whisper_language_per_clip():
    w = whisper(["es", "en"])
    results = w.transcribe_many([PCM, PCM, PCM], ["en", None, "es"])
    assert [lang for _, lang in results] == ["en", "en", "es"]  # None: picked among its languages
    assert w.model.calls == ["en", "en", "es"]
    assert w.transcribe_many([PCM]) == [("heard in en", "en")]   # no languages given: as before
    assert whisper(["es"]).transcribe_many([PCM]) == [("heard in es", "es")]


class FakeParakeet:
    def __init__(self, results):
        self.results = results

    def decode(self, audios):
        return self.results[: len(audios)]


def test_hybrid_rechecks_in_the_expected_language():
    w = whisper(["es"])
    hybrid = HybridTranscriber(FakeParakeet([("hello how are you", 0.0), ("hello how are you", 0.0),
                                             ("", 0.0)]), w)
    results = hybrid.transcribe_many([PCM, PCM, PCM], ["en", "es", None])
    assert results[0] == ("hello how are you", "auto")   # fine for an English server
    assert results[1] == ("heard in es", "es")           # no Spanish word: re-checked, in Spanish
    assert results[2] == ("heard in es", "es")           # empty: re-checked in Whisper's default
    assert w.model.calls == ["es", "es"]


class FakeEngine:
    def __init__(self):
        self.batches = []
        self.release = threading.Event()

    def transcribe_many(self, pcms, languages=None):
        self.release.wait(5)
        self.batches.append(list(languages))
        return [(f"text {i}", lang or "default") for i, lang in enumerate(languages)]


def test_batcher_passes_each_clips_language():
    engine = FakeEngine()
    batcher = TranscriptionBatcher(engine)
    first = batcher.submit(PCM, language="es")
    rest = [batcher.submit(PCM, language=lang) for lang in ("en", None, "es")]
    engine.release.set()
    assert first.result(5) == ("text 0", "es")
    assert [f.result(5)[1] for f in rest] == ["en", "default", "es"]
    # however the clips were grouped, each kept its own language, in order
    assert sum(engine.batches, []) == ["es", "en", None, "es"]


# ───────────────────────────── hallucinations & confidence ─────────────────────────────

def seg(text, logprob=-0.2, no_speech=0.01, compression=1.2):
    return (text, logprob, no_speech, compression)


def test_judge_keeps_clear_speech_with_its_confidence():
    text, confidence, no_speech, dropped = stt.judge([seg("Hola, ¿qué tal?", -0.25)], 1.2)
    assert (text, dropped) == ("Hola, ¿qué tal?", None)
    assert confidence == pytest.approx(-0.25) and no_speech == pytest.approx(0.01)


def test_judge_drops_subtitle_credits_always():
    for invented in ("Gracias por ver el video.", "Subtítulos realizados por la comunidad de Amara.org",
                     "Thanks for watching!"):
        assert stt.judge([seg(invented, -0.1)], 3.0)[3] == "hallucination", invented
    # a credit glued to real speech: the credit segment goes, the speech stays
    assert stt.judge([seg("vamos a jugar"), seg("Subtítulos por la comunidad de Amara.org")], 2.0)[0] == "vamos a jugar"


def test_judge_doubtful_gracias_needs_to_be_clearly_heard():
    assert stt.judge([seg("Gracias.", -0.2, 0.02)], 0.9)[3] is None              # really said
    assert stt.judge([seg("Gracias.", -0.2, 0.02)], 0.2)[3] == "hallucination"   # barely any speech
    assert stt.judge([seg("Gracias.", -0.2, 0.40)], 0.9)[3] == "hallucination"   # Whisper thinks silence
    assert stt.judge([seg("Thank you.", -0.6, 0.05)], 0.9)[3] == "hallucination"  # unsure of the words
    assert stt.judge([seg("Gracias por todo, amigo", -0.6, 0.3)], 0.9)[3] is None  # not a stock phrase


def test_judge_drops_silent_unsure_and_looping_segments():
    assert stt.judge([seg("eh bueno", -0.9, 0.8)], 1.0)[3] == "noise"
    assert stt.judge([seg("ja ja ja ja ja ja ja ja ja ja", -0.3, 0.0, 3.1)], 1.0)[3] == "noise"
    assert stt.judge([], 1.0) == ("", None, None, None)


def test_confidence_is_weighted_by_length():
    _, confidence, _, _ = stt.judge([seg("una frase larga y clara de verdad", -0.1), seg("eh", -0.9)], 2.0)
    assert -0.3 < confidence < -0.1


def test_whisper_skips_clips_without_speech(monkeypatch):
    monkeypatch.setattr(stt, "speech_seconds", lambda audio: 0.1)
    w = whisper(["es"])
    heard = w.transcribe_many([PCM])[0]
    assert heard == ("", "es") and heard.dropped == "noise"
    assert w.model.calls == []   # Whisper wasn't even asked


def test_heard_unpacks_like_before_and_carries_confidence():
    heard = whisper(["es"]).transcribe_many([PCM])[0]
    text, language = heard
    assert (text, language) == ("heard in es", "es")
    assert heard.confidence == pytest.approx(0.0) and heard.dropped is None
