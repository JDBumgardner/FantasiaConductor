"""Track display names.

A track's ``name`` stays the user's own word for it ("Kick"); the plugin is
shown alongside rather than baked in. Keeping them separate is what makes
clearing a plugin, renaming, and duplicating a track all behave — a name that
carries its plugin has to be rewritten on every one of those, and gets it wrong
whenever two of them happen at once.

Plain Python, like the rest of the model, so the label is computed the same way
for the UI, the agent and anything that comes later.
"""

from __future__ import annotations

SEP = " · "


def bare_name(name: str, plugin: str) -> str:
    """The track's own name, with a redundant plugin prefix removed.

    Names typed by hand tend to carry the plugin already ("Vital Arp",
    "Vital · Kick"). Stripping it here keeps the label from reading
    "Vital · Vital Arp", and makes the strip idempotent.
    """
    name = (name or "").strip()
    plugin = (plugin or "").strip()
    if not plugin:
        return name
    low, p = name.lower(), plugin.lower()
    if not low.startswith(p):
        return name
    rest = name[len(plugin):]
    # Only a whole word counts: "Vitality" is not a Vital track called "ity".
    if rest and not (rest[0].isspace() or rest[0] in SEP.strip() + "-_:"):
        return name
    rest = rest.lstrip()
    for lead in (SEP.strip(), "-", "_", ":"):
        if rest.startswith(lead):
            rest = rest[len(lead):].lstrip()
            break
    # "Vital" alone would strip to nothing — keep the name as the user left it.
    return rest or name


def track_label(track) -> str:  # noqa: ANN001
    """What to show for a track: ``Plugin · Name``, or just the name."""
    plugin = (getattr(track, "plugin", "") or "").strip()
    name = bare_name(getattr(track, "name", ""), plugin)
    return f"{plugin}{SEP}{name}" if plugin and name else (name or plugin)
