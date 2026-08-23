"""Project-wide tempo follow: rescale musical time when BPM changes.

Clip and note times are stored in seconds. Raising the tempo must shorten those
seconds (and vice versa) so the arrangement plays faster/slower like a DAW.

Audio clips remember how much of the source file they represent
(``source_duration``). That span stays put while ``duration`` is scaled; the
mixer warps the audio to fit.
"""

from __future__ import annotations


def source_span(clip) -> float:  # noqa: ANN001
    """Seconds of source audio this clip represents (pre-warp)."""
    span = float(getattr(clip, "source_duration", 0.0) or 0.0)
    return span if span > 0.0 else float(clip.duration)


def scale_timeline(project, factor: float) -> None:  # noqa: ANN001
    """Multiply every clip/note/fade time by ``factor``.

    ``factor = old_bpm / new_bpm``: 120→240 yields 0.5 (everything half as long).
    Audio ``source_offset`` / ``source_duration`` are file-native and not scaled.
    """
    if factor <= 0.0 or abs(factor - 1.0) < 1e-12:
        return
    for track in project.tracks:
        for clip in track.clips:
            if clip.source_path and float(getattr(clip, "source_duration", 0.0) or 0.0) <= 0.0:
                clip.source_duration = float(clip.duration)
            clip.start *= factor
            clip.duration *= factor
            clip.fade_in *= factor
            clip.fade_out *= factor
            for note in clip.notes:
                note.start *= factor
                note.duration *= factor
    project.loop_start = max(0.0, float(getattr(project, "loop_start", 0.0)) * factor)
    project.loop_end = max(project.loop_start + 0.05,
                           float(getattr(project, "loop_end", 8.0)) * factor)
