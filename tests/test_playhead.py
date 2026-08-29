"""The playhead slides between audio blocks instead of jumping per block."""

from types import SimpleNamespace as NS

import pytest

from fantasia_core.engine.playback import PlaybackEngine

SR = 44100
BLOCK = 8192          # ~186ms — one callback's worth


class _Stream:
    """Stands in for the PortAudio stream's clock."""

    def __init__(self, t=0.0):
        self.time = t


@pytest.fixture
def engine():
    e = PlaybackEngine(project=NS(duration=60.0, tempo=112.0, beats_per_bar=4,
                                 loop_enabled=False, tracks=[]),
                       pool=None, sample_rate=SR, block=BLOCK)
    e._playing = True
    e._stream = _Stream()
    return e


def _stamp(e, cursor_frames, dac_time):
    """What the callback records before rendering a block."""
    e._cursor = cursor_frames
    e._block_start = cursor_frames
    e._block_frames = BLOCK
    e._block_dac = dac_time


def test_playhead_moves_between_callbacks(engine):
    """Without this the cursor only changes once per block, so the playhead
    advances in steps of ~186ms — about a third of a beat at 112bpm."""
    _stamp(engine, 100 * SR, dac_time=10.0)
    seen = []
    for ms in range(0, 180, 20):          # UI polls ~33fps within one block
        engine._stream.time = 10.0 + ms / 1000.0
        seen.append(round(engine.playhead, 4))
    assert len(set(seen)) == len(seen), f"playhead stalled: {seen}"
    assert seen == sorted(seen)


def test_playhead_tracks_the_stream_clock_one_to_one(engine):
    _stamp(engine, 100 * SR, dac_time=10.0)
    engine._stream.time = 10.0
    at_start = engine.playhead
    engine._stream.time = 10.1
    assert engine.playhead == pytest.approx(at_start + 0.1, abs=1e-6)


def test_a_block_not_yet_heard_reads_behind_its_start(engine):
    """The callback fills a block the device has not played yet. Reporting its
    start would put the playhead ahead of the sound; walking back into the
    previous block is what keeps the cursor on the note being heard."""
    _stamp(engine, 100 * SR, dac_time=10.0)
    engine._stream.time = 9.95            # 50ms before this block is audible
    assert engine.playhead == pytest.approx(100.0 - 0.05, abs=1e-6)


def test_interpolation_stops_at_the_end_of_the_block(engine):
    """If the next callback is late the playhead must not run away."""
    _stamp(engine, 100 * SR, dac_time=10.0)
    engine._stream.time = 10.0 + 5.0      # callback never came
    assert engine.playhead == pytest.approx(100.0 + BLOCK / SR, abs=1e-6)


def test_it_never_reports_a_negative_position(engine):
    _stamp(engine, 0, dac_time=10.0)
    engine._stream.time = 9.0
    assert engine.playhead == 0.0


def test_falls_back_to_the_cursor_when_the_clock_is_unusable(engine):
    """Some backends report no DAC time; a closing stream has no clock."""
    _stamp(engine, 100 * SR, dac_time=0.0)          # no DAC time
    assert engine.playhead == pytest.approx(100.0)

    _stamp(engine, 100 * SR, dac_time=10.0)
    class _Dead:
        @property
        def time(self):
            raise RuntimeError("stream closed")
    engine._stream = _Dead()
    assert engine.playhead == pytest.approx(100.0)


def test_seeking_ignores_the_stale_block(engine):
    _stamp(engine, 100 * SR, dac_time=10.0)
    engine.set_playhead_seconds(20.0)
    engine._stream.time = 10.5
    assert engine.playhead == pytest.approx(20.0)


def test_a_stopped_engine_reports_the_cursor(engine):
    _stamp(engine, 100 * SR, dac_time=10.0)
    engine._playing = False
    engine._stream.time = 10.5
    assert engine.playhead == pytest.approx(100.0)
