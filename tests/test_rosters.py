"""Team rosters: the number -> name associations, saved once per team and linked to matches.

The interface is centred on these: the footage supplies the shirt *number* (per track, via the scan and the
stitching), and a roster supplies the *name* for it - typed once per squad, stored in the roster library, and
linked to each team of a match so a future game between the same teams starts with the names already on. What
these tests pin: the library round-trips and normalizes, a match remembers its links, and the merge rule never
lets a roster invent a number - it only names numbers something actually saw.
"""

from __future__ import annotations

import json

from soccer_analytics.analysis.identity import Appearance, identities_from_labels
from soccer_analytics.analysis.jerseys import merge_numbers
from soccer_analytics.analysis.library import MatchLibrary, MatchRecord
from soccer_analytics.analysis.rosters import RosterLibrary, clean_roster, roster_slug, team_rosters_for


# --------------------------------------------------------------------------------------------------------------
# The roster library
# --------------------------------------------------------------------------------------------------------------
def test_a_roster_round_trips_with_only_usable_associations(tmp_path) -> None:
    library = RosterLibrary(tmp_path)
    path = library.save("Tewksbury", {10: "Alice", "7": "  Bob  ", 0: "Nobody", 120: "Out of range", 11: ""})
    assert path.exists()
    assert library.load("Tewksbury") == {7: "Bob", 10: "Alice"}, "names stripped, junk and nameless dropped"
    assert library.names() == ["Tewksbury"]


def test_saving_under_an_existing_name_overwrites_it(tmp_path) -> None:
    library = RosterLibrary(tmp_path)
    library.save("Reds", {9: "Alice"})
    library.save("Reds", {9: "Alicia", 4: "Dana"})
    assert library.load("Reds") == {4: "Dana", 9: "Alicia"}
    assert library.names() == ["Reds"]


def test_rosters_are_found_by_case_insensitive_name(tmp_path) -> None:
    """A file is keyed by slug, so looking a roster up must not depend on how the name was capitalized."""
    library = RosterLibrary(tmp_path)
    library.save("Tewksbury", {10: "Alice"})
    assert library.load("tewksbury") == {10: "Alice"}
    assert roster_slug("Tewksbury") == roster_slug("tewksbury") == "tewksbury"


def test_a_missing_or_unreadable_roster_reads_as_empty(tmp_path) -> None:
    library = RosterLibrary(tmp_path)
    assert library.load("Nobody") == {}
    (tmp_path / "torn.json").write_text("{not json")
    assert library.load("torn") == {}
    assert library.names() == [], "an unreadable file is skipped, not fatal"


def test_a_roster_can_be_deleted(tmp_path) -> None:
    library = RosterLibrary(tmp_path)
    library.save("Reds", {9: "Alice"})
    assert library.delete("Reds") is True
    assert library.delete("Reds") is False
    assert library.names() == []


def test_clean_roster_normalizes_keys_and_drops_the_rest() -> None:
    assert clean_roster({"10": "Alice", "x": "Junk", 7: "Bob", 3: ""}) == {7: "Bob", 10: "Alice"}


# --------------------------------------------------------------------------------------------------------------
# The match's links
# --------------------------------------------------------------------------------------------------------------
def test_team_rosters_for_reads_only_linked_names(tmp_path) -> None:
    library = RosterLibrary(tmp_path)
    library.save("Reds", {9: "Alice"})
    resolved = team_rosters_for(library, ["Reds", ""])
    assert resolved == {0: {9: "Alice"}}, "an empty link contributes nothing"
    assert team_rosters_for(library, None) == {}
    # A link whose file has gone missing still resolves (to an empty map): the match loads, just without names.
    assert team_rosters_for(library, ["Gone", ""]) == {0: {}}


def test_match_record_remembers_the_linked_rosters(tmp_path) -> None:
    library = MatchLibrary(tmp_path)
    record = MatchRecord(match_id="2026-10-10_game", team_names=["Reds", "Blues"], team_rosters=["Reds 2026", ""])
    library.save(record)
    loaded = library.load("2026-10-10_game")
    assert loaded.team_rosters == ["Reds 2026", ""]
    assert loaded.team_roster(0) == "Reds 2026"
    assert loaded.team_roster(1) == ""
    assert loaded.team_roster(2) == "", "an out-of-range team has no roster rather than an IndexError"


def test_a_record_written_before_rosters_existed_has_none(tmp_path) -> None:
    """Old match.json files lack the key entirely; they must load with no links, not fail."""
    directory = tmp_path / "old_match"
    directory.mkdir()
    (directory / "match.json").write_text(json.dumps({"match_id": "old_match", "sources": [], "segments": []}))
    loaded = MatchLibrary(tmp_path).load("old_match")
    assert loaded.team_rosters == []
    assert loaded.team_roster(0) == ""


# --------------------------------------------------------------------------------------------------------------
# The merge rule
# --------------------------------------------------------------------------------------------------------------
def test_a_scanned_number_is_named_by_the_team_roster() -> None:
    auto = {7: {"number": 10, "confidence": 0.9}}
    merged = merge_numbers([7], auto=auto, team_of={7: 0}, team_rosters={0: {10: "Alice"}})
    assert merged[7] == {"number": 10, "name": "Alice", "source": "auto", "confidence": 0.9}


def test_a_manual_name_beats_the_team_roster() -> None:
    manual = {7: {"number": 10, "name": "Bob"}}
    merged = merge_numbers([7], manual=manual, team_of={7: 0}, team_rosters={0: {10: "Alice"}})
    assert merged[7]["name"] == "Bob" and merged[7]["source"] == "manual"


def test_a_manual_number_without_a_name_still_gets_the_roster_name() -> None:
    """Correcting a misread number must not lose the name the roster has for the corrected one."""
    manual = {7: {"number": 10, "name": ""}}
    merged = merge_numbers([7], manual=manual, team_of={7: 0}, team_rosters={0: {10: "Alice"}})
    assert merged[7] == {"number": 10, "name": "Alice", "source": "manual", "confidence": 1.0}


def test_a_roster_names_only_its_own_team() -> None:
    auto = {7: {"number": 9}, 8: {"number": 9}}
    merged = merge_numbers([7, 8], auto=auto, team_of={7: 0, 8: 1}, team_rosters={0: {9: "Alice"}})
    assert merged[7]["name"] == "Alice"
    assert merged[8]["name"] == "", "team 1 has no roster: its #9 stays unnamed"


def test_a_roster_cannot_claim_a_number_nobody_saw() -> None:
    """The number is evidence about the footage; a roster only names a number that is already known."""
    assert merge_numbers([7], team_of={7: 0}, team_rosters={0: {10: "Alice"}}) == {}


def test_roster_names_label_grouped_identities() -> None:
    """The promise of the feature end to end: two appearances of #10 become one player, named by the roster."""
    appearances = {
        4: Appearance(4, 0, first_t=0.0, last_t=20.0),
        9: Appearance(9, 0, first_t=60.0, last_t=90.0),
    }
    numbers = merge_numbers(
        [4, 9],
        auto={4: {"number": 10}, 9: {"number": 10}},
        team_of={4: 0, 9: 0},
        team_rosters={0: {10: "Alice"}},
    )
    identities = identities_from_labels(appearances, numbers)
    assert len(identities) == 1
    assert identities[0].members == (4, 9)
    assert identities[0].label == "#10 Alice"
