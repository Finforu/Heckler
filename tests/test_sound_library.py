import io

import numpy as np
import pytest
import soundfile as sf

from sound_library import (RATE, SILENT_DB, SoundLibrary, db_to_volume, effective_db, sound_name, trim_silence,
                           volume_to_db)


def wav_bytes(audio, rate=RATE, fmt="WAV") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format=fmt)
    return buf.getvalue()


def beep(seconds, amp=0.5, rate=RATE, lead=0.0, tail=0.0):
    t = np.arange(int(seconds * rate)) / rate
    tone = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return np.concatenate([np.zeros(int(lead * rate), np.float32), tone, np.zeros(int(tail * rate), np.float32)])


@pytest.fixture
def sounds(store, tmp_path):
    return SoundLibrary(store, tmp_path / "data")


def test_names():
    assert sound_name("Risa Malvada!") == "risa-malvada"
    assert sound_name("Bruh_2") == "bruh_2"
    for bad in ("", "!!!", "x" * 40):
        with pytest.raises(ValueError):
            sound_name(bad)


def test_trim_silence():
    audio = beep(1, lead=1, tail=2)
    trimmed = trim_silence(audio)
    assert 1.0 <= len(trimmed) / RATE <= 1.15
    assert len(trim_silence(np.zeros(RATE, np.float32))) == 0


def test_add_trims_levels_and_stores(store, sounds, tmp_path):
    data = wav_bytes(beep(2, amp=0.9, rate=44100, lead=0.5, tail=0.5), rate=44100, fmt="OGG")
    row = sounds.add(1, "Air Horn", data, created_by=42, filename="horn.ogg")
    assert row["name"] == "air-horn" and row["created_by"] == 42
    assert 2.0 <= row["duration_s"] <= 2.15
    path = tmp_path / "data" / "sounds" / "1" / f"{row['id']}.flac"
    assert path.is_file() and row["path"] == str(path.resolve())
    audio, rate = sf.read(path, dtype="float32")
    assert rate == RATE
    assert abs(np.sqrt(np.mean(audio ** 2)) - 0.1) < 0.01 and np.max(np.abs(audio)) < 1  # about speech level
    assert store.quota(1, 42, "sounds")[0] == 1

    with pytest.raises(ValueError, match="already"):
        sounds.add(1, "air horn", data, created_by=42)
    assert sounds.add(2, "air horn", data, created_by=42)  # names are per server


def test_limits(store, sounds):
    store.set_setting(1, "sounds.max_seconds", 3)
    row = sounds.add(1, "long", wav_bytes(beep(10)), created_by=1)
    assert row["duration_s"] == 3.0
    store.set_setting(1, "sounds.max_mb", 0.01)
    with pytest.raises(ValueError, match="MB"):
        sounds.add(1, "big", wav_bytes(beep(1)), created_by=1)
    store.set_setting(1, "sounds.max_mb", 5)
    with pytest.raises(ValueError):
        sounds.add(1, "silence", wav_bytes(np.zeros(RATE, np.float32)), created_by=1)
    with pytest.raises(ValueError):
        sounds.add(1, "junk", b"this is not audio at all" * 100, created_by=1)
    assert [s["name"] for s in store.list_sounds(1)] == ["long"]  # failures leave nothing behind


def test_pcm_and_delete(store, sounds):
    row = sounds.add(1, "beep", wav_bytes(beep(1)), created_by=1)
    pcm = sounds.pcm(row["id"])
    samples = len(pcm) // 4  # stereo s16
    assert abs(samples - row["duration_s"] * RATE) < RATE * 0.02
    assert sounds.pcm(row["id"]) is pcm  # cached
    store.update_sound(row["id"], gain_db=-6)
    quieter = np.frombuffer(sounds.pcm(row["id"]), np.int16)
    assert np.abs(quieter).max() < np.abs(np.frombuffer(pcm, np.int16)).max()

    path = sounds._disk_path(store.get_sound(row["id"])["path"])
    sounds.delete(row["id"])
    assert store.get_sound(row["id"]) is None and not path.exists()
    with pytest.raises(KeyError):
        sounds.pcm(row["id"])


def test_volume_percent_and_db():
    assert volume_to_db(100) == 0 and volume_to_db(0) == SILENT_DB
    assert volume_to_db(50) == pytest.approx(-6.02, abs=0.01)
    assert volume_to_db(200) == pytest.approx(6.02, abs=0.01)
    for percent in (0, 5, 50, 100, 150, 200, 400):
        assert db_to_volume(volume_to_db(percent)) == percent
    assert db_to_volume(None) == 100 and db_to_volume(-200) == 0
    for bad in (-1, 401):
        with pytest.raises(ValueError):
            volume_to_db(bad)


def test_server_cap():
    assert effective_db(6, 100) == 0          # a boost above the cap is held to it
    assert effective_db(-6, 100) == -6        # quieter sounds are left alone
    assert effective_db(6, 200) == 6
    assert effective_db(0, 50) == pytest.approx(-6.02, abs=0.01)
    assert effective_db(0, 0) == SILENT_DB
    assert effective_db(3, None) == 3


def peak(pcm: bytes) -> int:
    return int(np.abs(np.frombuffer(pcm, np.int16)).max())


def test_cap_applies_when_playing(store, sounds):
    row = sounds.add(1, "beep", wav_bytes(beep(1)), created_by=1)
    normal = peak(sounds.pcm(row["id"]))
    store.update_sound(row["id"], gain_db=volume_to_db(200))
    assert peak(sounds.pcm(row["id"])) == normal  # default cap: 100%
    store.set_setting(1, "sounds.max_volume", 200)
    loud = peak(sounds.pcm(row["id"]))
    assert loud > normal * 1.5
    store.set_setting(1, "sounds.max_volume", 50)
    assert peak(sounds.pcm(row["id"])) < normal * 0.6  # a lower cap turns down sounds set louder


def test_boost_never_clips(store, sounds):
    row = sounds.add(1, "spiky", wav_bytes(beep(1)), created_by=1)
    store.set_setting(1, "sounds.max_volume", 400)
    store.update_sound(row["id"], gain_db=volume_to_db(400))
    assert peak(sounds.pcm(row["id"])) <= 0.99 * 32767 + 1
