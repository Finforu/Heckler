"""DAVE (end-to-end encrypted voice) support for discord-ext-voice-recv.

Discord has required DAVE on every non-stage voice channel since March 2026.
discord.py 2.7 negotiates the session (via ``davey``) and encrypts what we
send, but voice-recv 0.5.2 only strips the transport layer: what it hands the
Opus decoder is still DAVE ciphertext, which decodes to static.

``install()`` patches voice-recv's ``PacketDecoder``:

* ``push_packet`` decrypts every frame exactly once, when it enters its
  speaker's decoder, which already knows the speaker's user id. RTP padding is
  stripped first (voice-recv ignores the padding bit, which would hide the
  DAVE marker), and non-Opus payloads are dropped.
* ``_decode_packet`` turns an Opus error on a single packet into silence.
  Unpatched, one bad packet kills voice-recv's router thread and with it all
  listening.

Frames we can't use become Opus silence rather than ``None``, because the
decoder's FEC path can't take ``None``.

It's also the cheapest place to ignore someone entirely: set ``ignore_user``
and their packets are dropped before any decrypting or decoding (the bot uses
it for music bots, which would otherwise be "listened to" non-stop).
"""
import logging
import time

import davey
from discord.ext.voice_recv.opus import PacketDecoder
from discord.opus import OpusError
from discord.ext.voice_recv.rtp import OPUS_SILENCE

log = logging.getLogger(__name__)

# Every DAVE-encrypted frame ends with this marker; silence frames and
# passthrough (non-E2EE) frames don't.
DAVE_MAGIC = b"\xfa\xfa"
OPUS_PAYLOAD_TYPE = 120
SILENT_PCM = b"\x00" * 3840  # one 20 ms frame of 48 kHz stereo 16-bit

_installed = False
_last_opus_warning = 0.0

# (voice_client, user_id) -> True to drop that user's audio. Set by the bot.
ignore_user = None


def _decrypt(decoder: PacketDecoder, data: bytes) -> bytes:
    vc = decoder.sink.voice_client
    session = vc._connection.dave_session if vc else None
    user_id = decoder._cached_id or (vc._get_id_from_ssrc(decoder.ssrc) if vc else None)

    if session is None or not session.ready or user_id is None:
        log.debug("Dropping DAVE frame for ssrc %s: session or user not ready", decoder.ssrc)
        return OPUS_SILENCE

    try:
        return session.decrypt(user_id, davey.MediaType.audio, data)
    except Exception as e:
        # Expected briefly around key rotations (someone joins or leaves).
        log.debug("DAVE decrypt failed for user %s: %s", user_id, e)
        return OPUS_SILENCE


def _strip_padding(data: bytes) -> bytes:
    # RFC 3550: the last padding byte holds the padding length, itself included.
    pad = data[-1]
    if 0 < pad <= len(data):
        return data[:-pad]
    return data


def _prepare(decoder: PacketDecoder, packet) -> None:
    data = packet.decrypted_data
    if not data or not hasattr(packet, "payload"):
        return  # fake/silence packets

    if packet.payload != OPUS_PAYLOAD_TYPE:
        log.debug("Dropping non-Opus payload type %s from ssrc %s", packet.payload, packet.ssrc)
        packet.decrypted_data = OPUS_SILENCE
        return

    if packet.padding:
        data = _strip_padding(data)

    if data.endswith(DAVE_MAGIC):
        data = _decrypt(decoder, data)

    packet.decrypted_data = data


def _describe(packet) -> str:
    data = packet.decrypted_data or b""
    return (
        f"ssrc={packet.ssrc} seq={packet.sequence} pt={getattr(packet, 'payload', '?')} "
        f"padding={getattr(packet, 'padding', '?')} extended={getattr(packet, 'extended', '?')} "
        f"len={len(data)} head={bytes(data[:8]).hex()} tail={bytes(data[-8:]).hex()}"
    )


def install() -> None:
    global _installed
    if _installed:
        return

    original_push = PacketDecoder.push_packet
    original_decode = PacketDecoder._decode_packet

    def push_packet(self: PacketDecoder, packet) -> None:
        if ignore_user is not None:
            vc = self.sink.voice_client
            user_id = self._cached_id or (vc._get_id_from_ssrc(self.ssrc) if vc else None)
            if vc is not None and user_id is not None and ignore_user(vc, user_id):
                return
        _prepare(self, packet)
        original_push(self, packet)

    def _decode_packet(self: PacketDecoder, packet):
        global _last_opus_warning
        try:
            return original_decode(self, packet)
        except OpusError as e:
            now = time.monotonic()
            if now - _last_opus_warning > 5:
                _last_opus_warning = now
                vc = self.sink.voice_client
                session = vc._connection.dave_session if vc else None
                log.warning(
                    "Undecodable voice packet (%s), using silence: %s | dave ready=%s",
                    e, _describe(packet), bool(session and session.ready),
                )
            return packet, SILENT_PCM

    PacketDecoder.push_packet = push_packet
    PacketDecoder._decode_packet = _decode_packet
    _installed = True
