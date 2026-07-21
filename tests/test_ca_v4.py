from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

import collect_ca_raw_cases as ca_collector
from collect_ca_raw_cases import (
    EXPECTED_KR_DISTRIBUTION,
    _chunk_funnel_rates,
    _truncate_to_checkpoint,
    build_shortlist,
    chunk_paths,
    collect_chunk,
    merge_chunks,
    output_paths,
    parser,
    read_kr_reference,
)


def strict_record(case_id: str, subtype: str, score: int = 90) -> dict[str, object]:
    return {
        "case_id": case_id,
        "strict_eligible": True,
        "claim_posture": "direct_tort_claim",
        "liability_basis": "non_contractual_tort",
        "court_system": "california_state",
        "court_level": "intermediate_appellate",
        "case_subtype": subtype,
        "factual_sufficiency_score": score,
        "main_opinion_confidence": "high",
        "governing_law_confidence": "high",
        "claim_posture_confidence": "high",
        "procedural_posture": "post_trial_appeal",
        "appellate_district": "Second Appellate District",
        "publication_status": "published",
        "decision_year": 2020,
        "death_involved": False,
        "physical_injury_involved": True,
        "property_damage_involved": False,
        "emotional_harm_involved": False,
        "human_qc_status": None,
        "human_qc_corrected_subtype": None,
        "human_qc_corrected_claim_posture": None,
        "human_qc_notes": None,
    }


def test_reference_file_distribution_is_read_not_inferred(tmp_path: Path):
    path = tmp_path / "kr.jsonl"
    rows = [{"case_subtype": subtype} for subtype, count in EXPECTED_KR_DISTRIBUTION.items() for _ in range(count)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    distribution, warnings = read_kr_reference(path)
    assert distribution == EXPECTED_KR_DISTRIBUTION
    assert warnings == []


def test_real_kr_reference_distribution_matches_documented_values():
    path = Path("outputs/raw/kr_v4/kr_cases_selected_50_final.jsonl")
    if not path.exists():
        pytest.skip("KR human-reviewed reference is not present")
    distribution, warnings = read_kr_reference(path)
    assert distribution == EXPECTED_KR_DISTRIBUTION
    assert warnings == []


def test_reference_mismatch_warns_and_file_remains_source_of_truth(tmp_path: Path):
    path = tmp_path / "kr.jsonl"
    path.write_text(json.dumps({"case_subtype": "traffic_accident"}) + "\n", encoding="utf-8")
    distribution, warnings = read_kr_reference(path)
    assert distribution == {"traffic_accident": 1}
    assert any("file_distribution_used" in warning for warning in warnings)


def test_shortlist_caps_each_subtype_at_25_and_leaves_human_fields_empty():
    pool = [strict_record(f"T{i:03}", "traffic_accident", 100 - i % 10) for i in range(60)]
    pool += [strict_record(f"M{i:03}", "medical_professional", 100 - i % 10) for i in range(60)]
    shortlist, _ = build_shortlist(
        pool, shortlist_count=100,
        kr_distribution={"traffic_accident": 10, "medical_professional": 9}, seed=42,
    )
    counts = Counter(row["case_subtype"] for row in shortlist)
    assert max(counts.values()) <= 25
    assert len(shortlist) == 50
    assert all(row["human_qc_status"] is None for row in shortlist)


def test_unclear_and_non_strict_records_never_enter_shortlist():
    pool = [strict_record("good", "other_tort"), strict_record("unclear", "unclear")]
    rejected = strict_record("rejected", "other_tort")
    rejected["strict_eligible"] = False
    shortlist, _ = build_shortlist(pool + [rejected], shortlist_count=100, kr_distribution={"other_tort": 1}, seed=42)
    assert [row["case_id"] for row in shortlist] == ["good"]


def test_v4_cli_has_no_final_selection_or_matching_options():
    options = parser().format_help()
    for forbidden in ("--select-final-sample", "--target-count", "--manual-qc-file", "--require-human-accept", "--match-kr-years", "--match-kr-lengths", "--relax-subtype-quota"):
        assert forbidden not in options


def test_output_contract_contains_no_ca_final_50_files(tmp_path: Path):
    names = {path.name for path in output_paths(tmp_path).values()}
    assert "ca_cases_selected_50_pre_qc.jsonl" not in names
    assert "ca_cases_selected_50_final.jsonl" not in names


def test_remote_loader_uses_bounded_streaming_not_parallel_materialization(monkeypatch):
    captured = {}

    def fake_load_dataset(*args, **kwargs):
        captured.update(kwargs)
        return iter([])

    monkeypatch.setattr(ca_collector, "load_dataset", fake_load_dataset)
    args = type("Args", (), {"local_arrow_dir": None, "source_loader": "datasets", "dataset": "source", "split": "train", "loader_batch_size": 16, "decision_date_from": "2000-01-01"})()
    assert list(ca_collector.iter_rows(args)) == []
    assert captured["streaming"] is True
    assert captured["batch_size"] == 16
    assert "num_proc" not in captured
    assert "keep_in_memory" not in captured
    assert ("court_type", "==", "SA") in captured["filters"]
    assert ("date_filed", ">=", date(2000, 1, 1)) in captured["filters"]


def test_filtered_api_loader_is_paginated_and_bounded(monkeypatch):
    calls = []

    class Response:
        def __init__(self, payload): self.payload = payload
        def raise_for_status(self): return None
        def json(self): return self.payload

    def fake_get(_endpoint, *, params, timeout):
        calls.append((dict(params), timeout))
        offset = params["offset"]
        rows = [{"row": {"id": index}} for index in range(offset, min(offset + params["length"], 3))]
        return Response({"num_rows_total": 3, "partial": True, "rows": rows})

    monkeypatch.setattr(ca_collector.requests, "get", fake_get)
    args = type("Args", (), {
        "local_arrow_dir": None, "source_loader": "datasets-server", "dataset": "source", "split": "train",
        "loader_page_size": 2, "decision_date_from": "2000-01-01",
    })()
    assert [row["id"] for row in ca_collector.iter_rows(args)] == [0, 1, 2]
    assert [call[0]["offset"] for call in calls] == [0, 2]
    assert all(call[0]["length"] == 2 for call in calls)
    assert args.metadata_scope_row_count == 3


def test_filtered_api_loader_applies_closed_chunk_dates(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self): return None
        def json(self): return {"num_rows_total": 0, "partial": False, "rows": []}

    def fake_get(_endpoint, *, params, timeout):
        captured.update(params); return Response()

    monkeypatch.setattr(ca_collector.requests, "get", fake_get)
    args = type("Args", (), {
        "local_arrow_dir": None, "source_loader": "datasets-server", "dataset": "source", "split": "train",
        "loader_page_size": 10, "decision_date_from": "2000-01-01", "year_min": 1980, "year_max": 1999,
        "source_start_offset": 0,
    })()
    assert list(ca_collector.iter_rows(args)) == []
    assert "1980-01-01" in captured["where"]
    assert "1999-12-31" in captured["where"]


def test_checkpoint_truncates_uncommitted_jsonl_tail(tmp_path: Path):
    path = tmp_path / "partial.jsonl"
    first = json.dumps({"id": 1}) + "\n"
    path.write_bytes((first + json.dumps({"id": 2}) + "\n").encode("utf-8"))
    checkpoint = {"partial_output_paths": {"court": str(path)}, "partial_output_offsets": {"court": len(first.encode("utf-8"))}}
    _truncate_to_checkpoint(checkpoint)
    assert path.read_bytes() == first.encode("utf-8")


def test_checkpoint_rejects_configuration_mismatch(tmp_path: Path):
    args = parser().parse_args(["--year-min", "2000", "--year-max", "2025", "--chunk-name", "2000_2025"])
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({
        "collection_version": ca_collector.COLLECTION_VERSION, "config_signature": "wrong",
    }), encoding="utf-8")
    with pytest.raises(RuntimeError):
        ca_collector._load_checkpoint(path, args)


def test_funnel_retention_rates_are_stage_specific():
    rates = _chunk_funnel_rates(Counter({
        "date_range_candidates": 100, "court_of_appeal_candidates": 80, "civil_candidates": 40,
        "broad_tort_candidates": 20, "direct_tort_candidates": 10,
        "california_state_law_candidates": 10, "full_main_opinion_candidates": 8,
        "factually_sufficient_candidates": 4,
    }))
    assert rates["court_of_appeal_retention_rate"] == 0.8
    assert rates["civil_retention_rate"] == 0.5
    assert rates["factual_sufficiency_retention_rate"] == 0.5


def test_unknown_principal_opinion_can_be_medium_confidence():
    text = (
        "FACTUAL BACKGROUND Plaintiff was a pedestrian. In 2020 defendant drove a vehicle and struck plaintiff in a collision. "
        "Plaintiff suffered bodily injury and medical expenses because of the collision. Defendant denied negligence and disputed causation. "
        "Plaintiff sued defendant for negligence damages. DISCUSSION duty breach causation damages."
    )
    row = {
        "id": "unknown-opinion", "case_name": "Smith v. Jones", "court_full_name": "California Court of Appeal",
        "court_jurisdiction": "California, CA", "court_type": "SA", "date_filed": "2020-01-01",
        "opinions": [{"type": "mystery-principal", "opinion_text": text}],
    }
    args = type("Args", (), {"min_opinion_chars": 20})()
    record = ca_collector.evaluate_row(row, args)
    assert record["main_opinion_type"] == "unknown"
    assert record["main_opinion_confidence"] == "medium"
    assert "no_valid_principal_opinion" not in record["exclusion_reasons"]


def test_direct_tort_with_incidental_insurance_background_is_not_coverage():
    text = (
        "Plaintiff sued defendant driver for negligence after defendant struck plaintiff in a 2020 collision. "
        "Plaintiff suffered bodily injury and damages because of the crash. Defendant denied causation and comparative fault. "
        "An insurance policy was mentioned only as background to the negligence claim."
    )
    result = ca_collector.classify_claim_posture(text, case_name="Victim v. Driver")
    assert result["claim_posture"] == "direct_tort_claim"


def test_chunk_paths_are_isolated_from_existing_merged_outputs(tmp_path: Path):
    paths = chunk_paths(tmp_path, "2000_2025")
    assert paths["strict"] == tmp_path / "chunks" / "ca_2000_2025_strict_eligible.jsonl"
    assert paths["summary"] == tmp_path / "chunks" / "ca_2000_2025_summary.json"


def test_chunk_collection_writes_checkpoint_and_completed_resume_is_idempotent(monkeypatch, tmp_path: Path):
    text = (
        "FACTUAL BACKGROUND Plaintiff was a pedestrian. In 2020 defendant drove a vehicle and struck plaintiff in a collision. "
        "Plaintiff suffered bodily injury and medical expenses because of the accident. Defendant denied negligence and disputed causation. "
        "Plaintiff sued defendant for negligence damages. DISCUSSION duty breach causation damages."
    )
    source_row = {
        "id": "chunk-one", "case_name": "Smith v. Jones", "court_full_name": "California Court of Appeal",
        "court_short_name": "California Court of Appeal", "court_jurisdiction": "California, CA", "court_type": "SA",
        "date_filed": "2020-01-01", "citations": ["1 Cal. App. 5th 1"], "opinions": [{"type": "010combined", "opinion_text": text}],
    }

    def fake_iter(args):
        if int(getattr(args, "source_start_offset", 0)) == 0:
            row = dict(source_row); row["_source_index"] = 0; yield row

    monkeypatch.setattr(ca_collector, "iter_rows", fake_iter)
    args = parser().parse_args([
        "--export-all-candidates", "--year-min", "2000", "--year-max", "2025", "--chunk-name", "2000_2025",
        "--checkpoint-every", "1", "--min-opinion-chars", "20", "--output-dir", str(tmp_path),
    ])
    assert collect_chunk(args) == 0
    paths = chunk_paths(tmp_path, "2000_2025")
    checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    assert checkpoint["completed"] is True
    assert len(paths["court"].read_text(encoding="utf-8").splitlines()) == 1
    assert len(paths["strict"].read_text(encoding="utf-8").splitlines()) == 1
    before = paths["strict"].read_bytes()
    args.resume = True
    assert collect_chunk(args) == 0
    assert paths["strict"].read_bytes() == before


def test_streaming_chunk_merge_removes_cross_chunk_duplicate_and_creates_no_final_50(tmp_path: Path):
    for chunk_name, year in (("1980_1999", 1999), ("2000_2025", 2000)):
        paths = chunk_paths(tmp_path, chunk_name)
        paths["court"].parent.mkdir(parents=True, exist_ok=True)
        row = strict_record("same-case", "traffic_accident")
        row.update({
            "source_record_id": "same-source", "decision_year": year, "publication_status": "published",
            "raw_text_sha256": "same-hash", "main_opinion_text": "full opinion", "collection_version": "test",
        })
        for key in ("court", "direct", "strict"):
            paths[key].write_text(json.dumps(row) + "\n", encoding="utf-8")
        paths["excluded"].write_text("", encoding="utf-8")
        paths["summary"].write_text(json.dumps({
            "period": chunk_name, "funnel_counts": {"total_scanned": 1, "strict_eligible_candidates": 1},
            "exclusion_reason_counts": {}, "peak_memory_bytes": 1024,
        }), encoding="utf-8")
        paths["checkpoint"].write_text(json.dumps({"completed": True, "chunk_name": chunk_name}), encoding="utf-8")

    reference = tmp_path / "kr.jsonl"
    reference.write_text("".join(
        json.dumps({"case_subtype": subtype}) + "\n"
        for subtype, count in EXPECTED_KR_DISTRIBUTION.items() for _ in range(count)
    ), encoding="utf-8")
    args = parser().parse_args([
        "--merge-chunks", "--output-dir", str(tmp_path), "--reference-kr-final", str(reference),
        "--manifest-output", str(tmp_path / "manifest.csv"), "--alignment-output", str(tmp_path / "alignment.csv"),
    ])
    assert merge_chunks(args) == 0
    merged = [json.loads(line) for line in output_paths(tmp_path)["strict"].read_text(encoding="utf-8").splitlines()]
    shortlist = [json.loads(line) for line in output_paths(tmp_path)["shortlist"].read_text(encoding="utf-8").splitlines()]
    assert len(merged) == 1
    assert len(shortlist) == 1
    assert not list(tmp_path.glob("ca_cases_selected_50*"))
    with pytest.raises(FileExistsError):
        merge_chunks(args)
