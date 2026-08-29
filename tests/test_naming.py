"""Track labels: the plugin is shown beside the name, never baked into it."""

from fantasia_core.document.model import Track
from fantasia_core.document.naming import bare_name, track_label


def _t(name, plugin=""):
    return Track(id="t1", name=name, plugin=plugin)


def test_label_pairs_plugin_with_the_users_own_name():
    assert track_label(_t("Kick", "Vital")) == "Vital · Kick"
    assert track_label(_t("Bass")) == "Bass"


def test_a_hand_typed_prefix_is_not_repeated():
    """Names typed before the badge existed carry the plugin already."""
    for name in ("Vital Arp", "Vital · Arp", "Vital_Arp", "vital arp"):
        assert track_label(_t(name, "Vital")).count("ital") == 1, name


def test_stripping_is_idempotent():
    once = bare_name("Vital Arp", "Vital")
    assert bare_name(once, "Vital") == once == "Arp"


def test_a_word_that_merely_starts_with_the_plugin_is_left_alone():
    assert bare_name("Vitality", "Vital") == "Vitality"
    assert bare_name("Vital", "Vital") == "Vital"  # nothing would be left


def test_clearing_the_plugin_leaves_a_clean_name():
    """The point of not baking it in: no rewrite needed when the plugin goes."""
    t = _t("Kick", "Vital")
    assert track_label(t) == "Vital · Kick"
    t.plugin = ""
    assert track_label(t) == "Kick"


def test_renaming_does_not_have_to_know_about_the_plugin():
    t = _t("Kick", "Vital")
    t.name = "Snare"
    assert track_label(t) == "Vital · Snare"
