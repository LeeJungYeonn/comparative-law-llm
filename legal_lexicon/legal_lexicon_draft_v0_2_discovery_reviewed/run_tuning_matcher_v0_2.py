#!/usr/bin/env python3
"""
Run the discovery-reviewed draft lexicon on the nested TUNING 10 only.

This script produces matcher evidence for human tuning review.
It does NOT modify the lexicon and does NOT access validation or confirmatory text.
"""

import argparse, csv, json, importlib.util
from pathlib import Path
from collections import Counter, defaultdict

def read_csv(path):
    with open(path,encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def load_matcher(path):
    spec=importlib.util.spec_from_file_location("draft_lexicon_matcher",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",required=True)
    p.add_argument("--nested-split-manifest",required=True)
    p.add_argument("--surface-forms",required=True)
    p.add_argument("--concepts",required=True)
    p.add_argument("--matcher",required=True)
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    split_rows=read_csv(a.nested_split_manifest)
    tuning={r["case_id"]:r for r in split_rows if r["nested_split"]=="tuning"}
    locked_validation={r["case_id"] for r in split_rows if r["nested_split"]=="validation"}
    assert len(tuning)==10
    assert Counter(r["origin_country"] for r in tuning.values())==Counter({"KR":5,"US":5})

    matcher=load_matcher(a.matcher)
    hit_rows=[]
    summaries=[]
    seen=set()

    with open(a.corpus,encoding="utf-8") as f:
        for line in f:
            rec=json.loads(line)
            cid=rec["case_id"]
            if cid not in tuning:
                continue
            assert rec["analysis_split"]=="development"
            assert cid not in locked_validation
            text=rec.get("main_opinion_text")
            assert isinstance(text,str) and text.strip()
            lang="ko" if rec["origin_country"]=="KR" else "en"
            hits=matcher.match_text(text,a.surface_forms,a.concepts,lang)
            seen.add(cid)

            cat=Counter(h["category"] for h in hits)
            summaries.append({
                "case_id":cid,
                "nested_split":"tuning",
                "origin_country":rec["origin_country"],
                "origin_state":rec.get("origin_state",""),
                "primary_domain":rec.get("primary_domain",""),
                "total_hits":len(hits),
                "doctrine_hits":cat["doctrine"],
                "remedy_hits":cat["remedy"],
                "procedure_hits":cat["procedure"],
                "party_arg_hits":cat["party_arg"],
                "zero_hits":str(len(hits)==0).lower(),
            })

            for i,h in enumerate(hits,1):
                hit_rows.append({
                    "case_id":cid,
                    "nested_split":"tuning",
                    "origin_country":rec["origin_country"],
                    "primary_domain":rec.get("primary_domain",""),
                    "hit_id":f"{cid}::H{i:04d}",
                    "category":h["category"],
                    "concept_id":h["concept_id"],
                    "concept_family":h["concept_family"],
                    "matched_text":h["matched_text"],
                    "start":h["start"],
                    "end":h["end"],
                    "context_before":text[max(0,h["start"]-140):h["start"]],
                    "context_after":text[h["end"]:h["end"]+140],
                    "review_tp_fp":"",
                    "review_category_correct":"",
                    "review_concept_correct":"",
                    "review_notes":"",
                })

    assert seen==set(tuning)

    hit_fields=[
        "case_id","nested_split","origin_country","primary_domain","hit_id","category","concept_id",
        "concept_family","matched_text","start","end","context_before","context_after",
        "review_tp_fp","review_category_correct","review_concept_correct","review_notes"
    ]
    with open(out/"tuning_matcher_hits_for_review.csv","w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=hit_fields);w.writeheader();w.writerows(hit_rows)

    summary_fields=["case_id","nested_split","origin_country","origin_state","primary_domain","total_hits",
                    "doctrine_hits","remedy_hits","procedure_hits","party_arg_hits","zero_hits"]
    with open(out/"tuning_matcher_summary.csv","w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=summary_fields);w.writeheader();w.writerows(sorted(summaries,key=lambda r:r["case_id"]))

    # FN audit template: human reviewer samples sentences/spans and records missed concepts.
    fn_fields=["case_id","origin_country","sample_id","text_span","start","end","expected_category",
               "expected_concept_family","expected_surface","matcher_missed","review_notes"]
    with open(out/"tuning_false_negative_audit.csv","w",encoding="utf-8-sig",newline="") as f:
        csv.DictWriter(f,fieldnames=fn_fields).writeheader()

    manifest={
        "stage":"tuning_matcher_run",
        "tuning_cases_used":10,
        "validation_cases_used":0,
        "confirmatory_cases_used":0,
        "lexicon_modified":False,
        "next_step":"human review of TP/FP hits and sampled FN audit; record any permitted changes in tuning_revision_log.csv",
    }
    (out/"tuning_run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print(f"tuning cases: {len(seen)}")
    print(f"matcher hits: {len(hit_rows)}")
    print(f"output: {out}")

if __name__=="__main__":
    main()
