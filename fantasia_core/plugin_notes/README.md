# Plugin notes

One JSON file per hosted plugin, recording what its parameters actually accept.

`plugin_params` tells you a parameter's *current* value; it does not tell you
which value **format** `set_plugin_param` will parse correctly. For Vital that
turned out to matter a lot — `"On"` sets a switch to Off and still returns
`ok: true` — so the answer is measured once and written down here instead of
being rediscovered by trial and error every session.

Each file records:

- `hosting_constraints` — how the plugin behaves inside this app (shared
  instances, re-render cost, known crashes)
- `value_format_rules` — raw 0-1 vs text, and where each applies
- `measured` — per-parameter, the value that worked and the form that failed
- `patches` — known-good settings lists, ready to replay

Always read back the `value` that `set_plugin_param` echoes; that is how these
entries were verified, and it is how a wrong format shows itself immediately.

## The full catalog

`vital_params.json` is the complete list — all 903 parameters, with display
name, init value, type, and group. It also carries the column that turns out to
matter most:

`json` — the field's name **inside the preset JSON**, which differs from the
`set_plugin_param` name for 256 of the 903 (`oscillator_1_switch` is
`osc_1_on`, `envelope_1_decay` is `env_1_decay`, `reverb_mix` is
`reverb_dry_wet`). Editing preset JSON under the parameter name writes a key
the plugin ignores and leaves the real field untouched, with no error.

## What is actually UI-only

Nothing, as it turns out. The parameter surface is knobs only — no modulation
source/destination, no wavetable choice, no sample slot, no LFO shape — but all
four live in the `.vital` JSON that `presets.splice_vital` rewrites, and all
four have been authored from scratch and verified against a live plugin. See
`capability_boundary` and `json_authoring_recipe` in `vital.json`.

The one habit worth keeping: a plugin will accept a preset blob, return
success, and go on using its previous patch. Snapshot the state back and re-read
it. For anything audible, render it and look at the spectrum.
