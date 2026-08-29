"""Claude tool-calling loop that drives the DAW through :class:`AgentTools`.

A manual agentic loop (Anthropic Messages API): call the model with the tool
definitions, execute any tool_use blocks via an injected ``execute_tool``
callback (the UI marshals that onto its own thread so bus edits stay on the UI
thread), feed results back, and repeat until the model stops calling tools.

The client is injectable so the loop is testable without the live API.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

# Default to Haiku — cheapest (~5x under Opus), fine for everyday edits. Override
# with FANTASIA_AGENT_MODEL (claude-sonnet-5 or claude-opus-5) for complex requests.
DEFAULT_MODEL = os.environ.get("FANTASIA_AGENT_MODEL", "claude-haiku-4-5")

# (input, output) USD per million tokens. Cache write = 1.25× input, read = 0.1× input.
_PRICING = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
}

SYSTEM = (
    "You are the assistant inside Fantasia Conductor, a music DAW. You compose and "
    "edit music by calling tools that mutate the project — the same commands the UI "
    "uses, so every edit is undoable.\n\n"
    "Workflow: first call get_project / list_tracks / list_clips to learn the current "
    "state and the ids you need, then make edits. New tracks use the built-in synth "
    "(a low-pass filtered, slightly detuned saw trio) by default. To write music: "
    "add_track, optionally set is_drum if you need a kit, add_clip (or a 4-bar MIDI "
    "clip) then write_midi with notes. Note times are seconds from the clip start; use "
    "get_project's seconds_per_beat to place notes on the grid. Drum pitches (on a drum "
    "track): 36 kick, 38 snare, 42 closed hat, 46 open hat, 49 crash. Keep velocities "
    "musical (accent beats louder).\n\n"
    "Designing sounds (synth tracks): new tracks already have a patch; reshape it with "
    "set_synth_patch(track_id, {params}). How the params sound:\n"
    "- osc1/osc2/osc3 waveform: sine=pure, saw=bright/buzzy, square=hollow/reedy, "
    "triangle=soft. mix 0=osc1 only … 1=equal detuned trio; detune ~0.1-0.2 semitones "
    "fattens the tone.\n"
    "- Amp envelope: attack (0=instant pluck, ~0.5s=slow pad swell), decay, sustain (0-1 held "
    "level), release (tail length).\n"
    "- Filter: cutoff Hz (low=dark/muffled, high=bright), resonance 0-1 (squelch/emphasis), "
    "env_amount Hz (opens the filter on attack for plucky/vocal sweeps). gain 0-1.\n"
    "Recipes: warm pad = saw+triangle, attack ~0.5, long release, cutoff ~1500, low resonance. "
    "Plucky bass = saw, fast attack, short decay, low sustain, env_amount ~2500. Gritty lead = "
    "detuned square+saw, high resonance, then add_fx a distortion. Layer add_fx reverb/delay/"
    "lowpass/highpass for space and character. The Master channel (id 'master') is a mix bus "
    "with no clips — put mix EQ/compression there. FX inserts have stable ids (list_fx); "
    "bypass_fx / move_fx / remove_fx address them by insert_id. Prefer get_eq / set_eq_band "
    "for the stock 8-band EQ (the same bands the user drags in the EQ view).\n\n"
    "Generating audio: for sounds the synth can't make (realistic drum hits, textures, risers, "
    "ambience, field recordings), use generate_audio(prompt, ...) — it runs MusicGen and fills a "
    "clip with a real waveform. It is slow, so use it sparingly and prefer synth/MIDI for musical "
    "parts.\n"
    "Finding sounds: before generating, use find_sound(query) to search the sound library for an "
    "existing sample that fits (e.g. 'warm pad', 'punchy kick'); place a match with "
    "add_sound(path, duration, track_id). Prefer a real sample over generation when one fits. "
    "Generated audio is automatically added to the library, so a sound you generate once can be "
    "found again with find_sound.\n"
    "Isolating instruments: separate_stems(clip_id) splits an audio clip into drums/bass/vocals/other "
    "stems on new tracks — use it to extract or remove an instrument from imported or generated audio.\n"
    "Time-stretch: stretch_clip(clip_id, factor) changes an audio clip's length WITHOUT changing pitch "
    "(factor 2=half speed, 0.5=double speed); stretch_clip_to_bars(clip_id, bars) fits a loop to N bars at the "
    "tempo. Pitch shifting (length-preserving) is set_clip(pitch_semitones=…).\n"
    "Voice: speak(text, voice) synthesizes spoken vocals (Kokoro TTS) into a clip — use it for lyrics, "
    "spoken word, or vocal samples. Voices include af_heart/am_michael (American), bf_emma/bm_george (British).\n"
    "Singing: sing(clip_id, lyrics, voice) turns a MIDI melody into a sung vocal (one syllable per note; "
    "hyphen-split words). Write/draw a melody in a MIDI clip first, then sing it — the vocal lands on a new track.\n"
    "Vocal IN TIME with the song: prefer sing_melody(notes, lyrics, start_beat, voice) — you compose the vocal "
    "melody with note times in BEATS so it is automatically locked to the tempo grid. Recipe: (1) call get_project "
    "for tempo + beats_per_bar; (2) inspect existing MIDI clips (get_clip_notes) to pick a key that fits; (3) write "
    "notes as {pitch, beat, beats} where beat is the start in beats from start_beat and a bar = beats_per_bar beats; "
    "(4) give one lyric syllable per note (hyphen-split words). start_beat=0 is bar 1, start_beat=16 is bar 5 in 4/4. "
    "The vocal renders on a new track exactly in sync with the arrangement.\n"
    "Vocal FX: vocal_fx(clip_id, effect) polishes a vocal audio clip — autotune (key+scale), harmony (adds a "
    "voice a given interval up on a new track), formant_up/down, deess, double. Formant-preserving (WORLD).\n\n"
    "TIMING — think in bars and beats, never in seconds. add_clip takes bar (1-based measure) "
    "and bars (length); notes take bar + beat (1-based within the bar, fractional: 2.5 is the "
    "'and' of 2) + beats (length: 1=quarter, 0.5=eighth, 0.25=sixteenth). The app converts to "
    "seconds using the project tempo and rebases against the clip, so a note at bar 5 beat 1 "
    "lands on bar 5 beat 1 no matter where its clip starts. Write a bar-4 backbeat as "
    "{pitch:38, bar:4, beat:2, beats:0.5}. Notes placed outside their clip are rejected with the "
    "clip's bar range — widen the clip or move the notes rather than guessing.\n\n"
    "Be decisive and act rather than asking for confirmation on ordinary edits. When done, "
    "give a one- or two-sentence summary of what you changed."
)


class AgentSession:
    def __init__(self, tools, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 max_steps: int = 16, client=None) -> None:
        self.tools = tools
        self.model = model
        self._api_key = api_key
        self.max_steps = max_steps
        self._client = client
        self.messages: list = []  # persistent conversation
        self.session_cost = 0.0   # running USD for this app session
        self.session_tokens = 0

    # ---- availability / client ------------------------------------------
    @staticmethod
    def anthropic_available() -> bool:
        try:
            import anthropic  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def available(self) -> bool:
        if self._client is not None:
            return True
        if not self.anthropic_available():
            return False
        return bool(self._api_key or os.environ.get("ANTHROPIC_API_KEY")
                    or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key) if self._api_key \
                else anthropic.Anthropic()
        return self._client

    def reset(self) -> None:
        """Start a fresh conversation (clears history → cheaper next request)."""
        self.messages = []

    def _mark_history_cache(self) -> None:
        """Move a cache breakpoint to the newest message so the whole conversation
        prefix is served from cache (~0.1x) instead of re-billed at full input
        price on every API call. Only touches dict blocks we authored (assistant
        turns are SDK objects and never carry markers)."""
        for msg in self.messages:  # drop stale breakpoints (max 4 allowed total)
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)
        last = self.messages[-1] if self.messages else None
        if not isinstance(last, dict):
            return
        content = last.get("content")
        if isinstance(content, str):
            last["content"] = [{"type": "text", "text": content,
                                "cache_control": {"type": "ephemeral"}}]
        elif isinstance(content, list) and content and isinstance(content[-1], dict):
            content[-1]["cache_control"] = {"type": "ephemeral"}

    @staticmethod
    def _cached_tools(defs: list) -> list:
        """Add a cache breakpoint at the last tool so the (large, static) tool
        schemas + system prompt are billed at ~0.1x on subsequent turns."""
        if not defs:
            return defs
        out = [dict(d) for d in defs]
        out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
        return out

    def _account(self, resp) -> dict:
        """Tally tokens + cost from a response's usage; return a delta summary."""
        u = getattr(resp, "usage", None)
        if u is None:
            return {}
        inp = getattr(u, "input_tokens", 0) or 0
        out = getattr(u, "output_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        rate = _PRICING.get(self.model)
        cost = 0.0
        if rate:
            cin, cout = rate
            cost = (inp * cin + cw * cin * 1.25 + cr * cin * 0.10 + out * cout) / 1_000_000
        self.session_cost += cost
        self.session_tokens += inp + out + cw + cr
        return {"cost": cost, "cumulative_cost": self.session_cost,
                "input": inp, "output": out, "cache_read": cr, "cache_write": cw,
                "cumulative_tokens": self.session_tokens, "model": self.model}

    # ---- run -------------------------------------------------------------
    def run(self, user_message: str,
            on_text: Callable[[str], None],
            execute_tool: Callable[[str, dict], object],
            on_usage: Optional[Callable[[dict], None]] = None) -> str:
        client = self._ensure_client()
        defs = self._cached_tools(self.tools.definitions())
        system = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
        self.messages.append({"role": "user", "content": user_message})

        final_text = ""
        for _ in range(self.max_steps):
            self._mark_history_cache()  # cache the conversation prefix too
            resp = client.messages.create(
                model=self.model, max_tokens=4096, system=system,
                tools=defs, messages=self.messages,
            )
            if on_usage is not None:
                on_usage(self._account(resp))
            if getattr(resp, "stop_reason", None) == "refusal":
                on_text("(The request was declined.)")
                return "(declined)"

            self.messages.append({"role": "assistant", "content": resp.content})
            text = "".join(b.text for b in resp.content if b.type == "text")
            if text:
                on_text(text)
                final_text = text

            if resp.stop_reason == "pause_turn":
                continue  # server-tool loop paused — resend to resume

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                return final_text  # end_turn

            results = []
            for tu in tool_uses:
                try:
                    out = execute_tool(tu.name, dict(tu.input))
                    results.append({"type": "tool_result", "tool_use_id": tu.id,
                                    "content": json.dumps(out)})
                except Exception as exc:  # noqa: BLE001
                    results.append({"type": "tool_result", "tool_use_id": tu.id,
                                    "content": json.dumps({"error": str(exc)}), "is_error": True})
            self.messages.append({"role": "user", "content": results})

        return final_text or "(stopped after the step limit)"
