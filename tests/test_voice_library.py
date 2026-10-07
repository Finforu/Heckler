"""The voice library with a fake model: no torch, no GPU. Audio is synthetic."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import voice_library as vl
from store import Store
from voice_library import RATE, VoiceLibrary, VoiceNotReady


class FakePrompt:
    def __init__(self, ref_text, samples):
        self.ref_text, self.samples = ref_text, samples

    def save(self, path):
        Path(path).write_text(json.dumps({"ref_text": self.ref_text, "samples": self.samples}))


def fake_load(path, map_location="cpu"):
    data = json.loads(Path(path).read_text())
    return FakePrompt(data["ref_text"], data["samples"])


class FakeModel:
    sampling_rate = RATE
    device = "cpu"
    name_or_path = "fake/omnivoice"

    def __init__(self):
        self.prompts = []    # (samples, ref_text)
        self.generated = []  # generate() kwargs

    def create_voice_clone_prompt(self, ref_audio, ref_text=None):
        wave, rate = ref_audio
        assert rate == RATE and wave.ndim == 2 and wave.shape[0] == 1
        self.prompts.append((wave.shape[1], ref_text))
        return FakePrompt(ref_text, int(wave.shape[1]))

    def generate(self, text, language=None, num_step=None, voice_clone_prompt=None, instruct=None, speed=None):
        self.generated.append(dict(text=text, language=language, num_step=num_step, speed=speed,
                                   prompt=voice_clone_prompt, instruct=instruct))
        return [np.full(RATE // 2, 0.25, dtype=np.float32)]


def tone(seconds, freq=220.0, amp=0.3):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def transcribed():
    return []


@pytest.fixture
def make_library(store, tmp_path, model, transcribed):
    def make(**kwargs):
        library = VoiceLibrary(store, tmp_path / "data", language="es", **kwargs)

        def transcribe(audio16):
            transcribed.append(len(audio16))
            return "hola esto es una prueba"

        library.attach(model, transcribe, load_prompt=fake_load)
        return library

    return make


@pytest.fixture
def library(make_library):
    return make_library()


def wav_file(path: Path, audio, rate=RATE) -> Path:
    sf.write(path, audio, rate)
    return path


# ------------------------------------------------------------ clones
def test_ingest_transcribe_build(store, library, model, transcribed, tmp_path):
    voice_id = store.add_voice("pirate", guild_id=1)
    source = wav_file(tmp_path / "Pirate.WAV", tone(20, amp=0.02), rate=48000)
    row = library.ingest(voice_id, source)
    folder = library.folder(voice_id)
    assert (folder / "source.wav").is_file() and row["status"] == "draft"
    ref, rate = sf.read(folder / "ref.wav", dtype="float32")
    assert rate == RATE and ref.ndim == 1
    assert len(ref) <= vl.REF_MAX_SECONDS * RATE                       # capped
    assert abs(np.sqrt(np.mean(ref ** 2)) - vl.REF_RMS) < 0.01        # leveled (it was quiet)
    assert Path(row["ref_path"]).name == "ref.wav"

    assert library.transcribe_ref(voice_id) == "hola esto es una prueba"
    assert transcribed and transcribed[0] == pytest.approx(len(ref) * 16000 / RATE, abs=2)
    assert library.build_due(voice_id)
    assert library.build(voice_id) is True
    row = store.get_voice(voice_id)
    assert row["status"] == "ready" and row["error"] is None and row["prompt_hash"]
    assert (folder / "prompt.pt").is_file() and model.prompts == [(len(ref), "hola esto es una prueba")]
    assert not library.build_due(voice_id)

    # unchanged: skipped; a new transcript: rebuilt, new hash
    assert library.build(voice_id) is False and len(model.prompts) == 1
    first_hash = row["prompt_hash"]
    store.update_voice(voice_id, ref_text="otra cosa")
    assert library.build(voice_id) is True and len(model.prompts) == 2
    assert store.get_voice(voice_id)["prompt_hash"] != first_hash
    assert len(transcribed) == 1  # a typed (or kept) transcript is never redone


def test_ingest_selection_and_raw_audio(store, library):
    voice_id = store.add_voice("v", guild_id=1)
    library.ingest(voice_id, tone(30), 48000, start_s=2, end_s=5)
    ref, _ = sf.read(library.folder(voice_id) / "ref.wav")
    assert len(ref) == 3 * RATE
    library.ingest(voice_id, tone(30), RATE, start_s=0)
    ref, _ = sf.read(library.folder(voice_id) / "ref.wav")
    assert len(ref) == vl.REF_MAX_SECONDS * RATE
    with pytest.raises(ValueError):
        library.ingest(voice_id, tone(30), RATE, start_s=1, end_s=1.1)
    with pytest.raises(ValueError):
        library.ingest(voice_id, tone(3))  # raw audio needs its rate


def test_a_failed_build_is_recorded(store, library, model):
    voice_id = store.add_voice("v", guild_id=1)
    library.ingest(voice_id, tone(5), RATE)

    def boom(*a, **k):
        raise MemoryError("CUDA out of memory")

    model.create_voice_clone_prompt = boom
    with pytest.raises(MemoryError):
        library.build(voice_id)
    row = store.get_voice(voice_id)
    assert row["status"] == "failed" and "out of memory" in row["error"]


def test_a_packed_voice_builds_from_its_source(store, library, tmp_path):
    source = wav_file(tmp_path / "from-pack.wav", tone(4))
    voice_id = store.add_voice("packed", guild_id=1, source_path=str(source), status="queued")
    assert library.pending_builds() == [voice_id]
    assert library.build(voice_id)
    assert store.get_voice(voice_id)["status"] == "ready" and library.pending_builds() == []


# ------------------------------------------------------------ speaking and the cache
def test_disk_cache_survives_a_restart(store, make_library, model):
    library = make_library()
    voice_id = store.add_voice("v", guild_id=1, gain_db=-6)
    library.ingest(voice_id, tone(5), RATE)
    library.build(voice_id)

    assert library.cached("hola", voice_id) is None
    audio = library.speak("hola", voice_id)
    assert audio.dtype == np.float32 and len(audio) == RATE // 2
    assert np.allclose(audio, 0.25 * 10 ** (-6 / 20), atol=1e-3)  # gain applied on the way out
    assert len(model.generated) == 1
    call = model.generated[0]
    assert call["language"] == "es" and call["num_step"] == vl.DEFAULT_NUM_STEP and call["instruct"] is None
    assert call["prompt"].samples > 0

    library.speak("hola", voice_id)
    assert len(model.generated) == 1  # from the cache

    restarted = make_library()  # a new process, same folder
    again = restarted.speak("hola", voice_id)
    assert np.allclose(again, audio, atol=1e-3) and len(model.generated) == 1
    assert restarted.cached("hola", voice_id) is not None

    # other settings are other lines
    restarted.speak("hola", voice_id, num_step=8)
    restarted.speak("hola", voice_id, speed=1.2)
    assert [g["num_step"] for g in model.generated[1:]] == [8, vl.DEFAULT_NUM_STEP]
    assert model.generated[2]["speed"] == 1.2

    # a rebuild changes the voice: its old lines go
    store.update_voice(voice_id, ref_text="nuevo texto")
    restarted.build(voice_id)
    assert restarted.cached("hola", voice_id) is None


def test_language_per_call(store, library, model):
    voice_id = store.add_voice("v", guild_id=1, kind="designed", instruct="male")
    library.speak("hola", voice_id)                  # the library default ("es" in these tests)
    library.speak("hola", voice_id, language="en")   # an English server
    library.speak("hola", voice_id, language="en")   # cached
    assert [g["language"] for g in model.generated] == ["es", "en"]
    assert library.cached("hola", voice_id, language="en") is not None
    assert library.cache_path("hola", voice_id, language="en") != library.cache_path("hola", voice_id)
    store.update_voice(voice_id, language="fr")      # a voice with its own language wins
    library.speak("hola", voice_id, language="en")
    assert model.generated[-1]["language"] == "fr"
    assert library.cache_path("hola", voice_id, language="en") == library.cache_path("hola", voice_id)


def test_designed_voices(store, library, model):
    voice_id = store.add_voice("narrator", guild_id=1, kind="designed", instruct="deep voice", speed=0.9)
    assert library.build_due(voice_id)
    assert library.build(voice_id)
    assert store.get_voice(voice_id)["status"] == "ready" and not (library.folder(voice_id) / "prompt.pt").exists()
    library.speak("hola", voice_id)
    library.speak("hola", voice_id)
    assert len(model.generated) == 1 and model.generated[0]["instruct"] == "deep voice"
    assert model.generated[0]["speed"] == 0.9 and model.generated[0]["prompt"] is None
    store.update_voice(voice_id, instruct="high voice")  # a new identity: a new line
    library.speak("hola", voice_id)
    assert len(model.generated) == 2

    empty = store.add_voice("empty", guild_id=1, kind="designed")
    with pytest.raises(ValueError):
        library.build(empty)
    assert store.get_voice(empty)["status"] == "failed"


def test_unbuilt_voices_and_missing_bot_voice(store, library):
    voice_id = store.add_voice("v", guild_id=1)
    with pytest.raises(VoiceNotReady):
        library.speak("hola", voice_id)
    with pytest.raises(VoiceNotReady):
        library.speak("hola")  # no bot voice yet
    assert library.cached("hola", voice_id) is None


def test_cache_eviction(store, library):
    voice_id = store.add_voice("v", guild_id=1, kind="designed", instruct="x")
    for i in range(10):
        library.speak(f"line {i}", voice_id)
    size = library.cache_size()
    assert size > 0
    per_file = size / 10
    library.cached("line 0", voice_id)  # just used: stays
    paths = {i: library.cache_path(f"line {i}", voice_id) for i in range(10)}
    import os
    for i, path in paths.items():  # deterministic ages: 0 newest, then 9, 8, ...
        os.utime(path, (1000 + (100 if i == 0 else i), 1000 + (100 if i == 0 else i)))

    store.set_setting(0, "cache.tts_mb", (per_file * 5) / 1024 / 1024)
    library.speak("line 10", voice_id)  # over the limit: the oldest go
    left = {i for i, p in paths.items() if p.exists()}
    assert 0 in left and 1 not in left and library.cache_size() <= per_file * 5
    assert library.cache_path("line 10", voice_id).exists()

    library.clear_cache(voice_id)
    assert library.cache_size() == 0


# ------------------------------------------------------------ the bot's voice
def test_ensure_bot_voice(store, library, model, tmp_path):
    store.set_setting(0, "bot.name", "Robo")
    mp3 = wav_file(tmp_path / "bot voice.wav", tone(8))
    old_prompt = tmp_path / "models" / "bot-voice-123.pt"
    old_prompt.parent.mkdir()
    FakePrompt("hola soy el bot", 99).save(old_prompt)

    voice_id = library.ensure_bot_voice(mp3, prompt_file=old_prompt)
    row = store.get_voice(voice_id)
    assert row["name"] == "Robo" and row["guild_id"] is None and row["kind"] == "clone"
    assert library.bot_voice_id() == voice_id == store.get_setting(0, "voice.bot")
    assert row["status"] == "ready" and not library.build_due(voice_id)
    assert model.prompts == []  # reused, not rebuilt

    library.speak("hola")  # None: the bot's voice
    assert model.generated[0]["prompt"].samples == 99
    assert store.get_voice(voice_id)["ref_text"] == "hola soy el bot"  # learned from the prompt

    assert library.ensure_bot_voice(mp3) == voice_id  # same file: nothing to do
    assert store.get_voice(voice_id)["status"] == "ready"

    wav_file(mp3, tone(9, freq=330))  # the file changed
    import os
    os.utime(mp3, (5000, 5000))
    assert library.ensure_bot_voice(mp3) == voice_id
    assert library.build_due(voice_id)
    library.build(voice_id)
    assert len(model.prompts) == 1 and store.get_voice(voice_id)["status"] == "ready"


# ------------------------------------------------------------ people's own voices
USER = 4242


def clip(seconds=2.0):
    return tone(seconds, freq=200 + seconds * 10)


def test_no_consent_no_files(store, library, tmp_path):
    assert library.update_speaker(USER, clip(), "hola que tal amigo") is False
    assert store.find_voices(kind="speaker") == []
    assert not (tmp_path / "data" / "voices").exists()
    assert library.speaker_voice(1, USER) is None


def test_speaker_builds_once_at_the_threshold(store, library, model):
    store.set_consent(USER, "accepted")
    assert library.update_speaker(USER, clip(0.5), "muy corto") is False         # under 1 s: dropped
    assert library.update_speaker(USER, clip(), "hola") is False                 # one word: dropped
    due = [library.update_speaker(USER, clip(), f"frase numero {i}") for i in range(6)]
    assert due == [False, False, False, True, False, False]  # 8 s reached once
    row = library.speaker_voice(1, USER, ready_only=False)
    assert row["status"] == "queued" and row["owner_user_id"] == USER and row["guild_id"] is None
    assert library.speaker_voice(1, USER) is None  # not built yet
    assert library.pending_builds() == [row["id"]]
    assert library._clip_seconds(row["id"]) == pytest.approx(10.0, abs=0.01)  # newest ~10 s kept

    library.build(row["id"])
    built = library.speaker_voice(1, USER)
    assert built["status"] == "ready" and built["ref_text"] == "frase numero 4 frase numero 5"  # 6 s, pauses included
    assert model.prompts[0][0] <= vl.SPEAKER_REF_SECONDS * RATE
    assert library.update_speaker(USER, clip(), "una frase mas") is False  # no rebuild per clip

    assert library.request_speaker_rebuild(USER) == row["id"]
    assert store.get_voice(row["id"])["status"] == "queued"

    store.set_consent(USER, "revoked")
    assert library.speaker_voice(1, USER) is None
    assert library.update_speaker(USER, clip(), "ya no se guarda") is False


def test_delete_speaker_removes_everything(store, library, model):
    store.set_consent(USER, "accepted")
    for i in range(5):
        library.update_speaker(USER, clip(), f"frase numero {i}")
    voice_id = library.speaker_voice(1, USER, ready_only=False)["id"]
    library.build(voice_id)
    library.speak("hola", voice_id)
    assert library.cache_path("hola", voice_id).exists()

    assert library.delete_speaker(USER) == 1
    assert store.get_voice(voice_id) is None
    assert not library.folder(voice_id).exists()
    assert not (library.cache_dir / str(voice_id)).exists()
    assert library.speaker_voice(1, USER, ready_only=False) is None


def test_delete_voice(store, library):
    voice_id = store.add_voice("v", guild_id=1, kind="designed", instruct="x")
    library.speak("hola", voice_id)
    library.delete_voice(voice_id)
    assert store.get_voice(voice_id) is None and library.cache_size() == 0


def test_to_discord_pcm():
    pcm = vl.to_discord_pcm(np.zeros(RATE, dtype=np.float32))
    assert len(pcm) == 48000 * 2 * 2  # 1 s of 48 kHz stereo s16
