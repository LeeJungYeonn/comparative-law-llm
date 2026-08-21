#!/usr/bin/env python3
import csv, re, json
from pathlib import Path
from collections import defaultdict

CATEGORY_ORDER = ["remedy","doctrine","procedure","party_arg"]

def load_rows(path):
    with open(path,encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def overlaps(a,b):
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])

def compile_pattern(row):
    if row["match_type"]=="regex":
        # Intentionally NOT DOTALL: party-argument patterns must not cross line boundaries.
        return re.compile(row["surface_form"], re.IGNORECASE)

    pat = re.escape(row["surface_form"]).replace(r"\ ", r"\s+")

    # English literal expressions require whole-token boundaries to prevent
    # false matches such as affirm -> affirmative or answer -> answered.
    if row["language"] == "en":
        pat = rf"(?<![A-Za-z0-9_]){pat}(?![A-Za-z0-9_])"

    # Korean is intentionally left without word boundaries because case particles
    # and endings may attach directly to legal terms.
    return re.compile(pat, re.IGNORECASE)

def generate_hits(text, lex_rows, concept_meta):
    out=[]
    for row in lex_rows:
        c=concept_meta[row["concept_id"]]
        for m in compile_pattern(row).finditer(text):
            out.append({
                "surface_id":row["surface_id"],
                "concept_id":row["concept_id"],
                "concept_family":c["concept_family"],
                "category":c["category"],
                "jurisdiction_scope":c["jurisdiction_scope"],
                "orientation_layer":c["orientation_layer"],
                "language":row["language"],
                "matched_text":text[m.start():m.end()],
                "start":m.start(),
                "end":m.end(),
                "span_length":m.end()-m.start(),
                "primary_include":int(row["primary_include"]),
                "orientation_include":int(row["orientation_include"]),
                "ambiguity":row["ambiguity"],
                "provenance":row["provenance"],
            })
    return out

def resolve_exact_span_collisions(hits):
    """
    If multiple lexicon rows produce the same exact span in the same category,
    collapse them before longest-match selection.

    Cross-jurisdiction duplicates sharing one functional concept family are
    treated as SHARED for the hit log and cannot contribute to orientation.
    This prevents arbitrary surface_id tie-breaking (e.g., 'product liability',
    'non-economic damages', 'remand', 'reverse').
    """
    groups=defaultdict(list)
    for h in hits:
        key=(h["start"],h["end"],h["category"],h["matched_text"].casefold())
        groups[key].append(h)

    resolved=[]
    for _, group in groups.items():
        if len(group)==1:
            h=group[0].copy()
            h["candidate_concept_ids"]=[h["concept_id"]]
            resolved.append(h)
            continue

        families={h["concept_family"] for h in group}
        scopes={h["jurisdiction_scope"] for h in group}

        if len(families)==1:
            family=next(iter(families))
            # Choose a stable representative for non-jurisdiction metadata,
            # but explicitly mark the collision when scopes differ.
            rep=sorted(group,key=lambda h:h["surface_id"])[0].copy()
            rep["candidate_concept_ids"]=sorted({h["concept_id"] for h in group})
            rep["primary_include"]=max(h["primary_include"] for h in group)
            rep["ambiguity"]="cross_jurisdiction_duplicate" if len(scopes)>1 else rep["ambiguity"]

            if len(scopes)>1:
                rep["concept_id"]=f"SHARED::{family}"
                rep["jurisdiction_scope"]="SHARED"
                rep["orientation_layer"]="shared"
                rep["orientation_include"]=0
            else:
                # Same-jurisdiction synonym/duplicate: keep stable representative,
                # but never double-count the same exact span.
                rep["orientation_include"]=max(h["orientation_include"] for h in group)
            resolved.append(rep)
        else:
            # Different functional families on exactly the same span are kept
            # as an explicit conflict and excluded from orientation until reviewed.
            rep=sorted(group,key=lambda h:h["surface_id"])[0].copy()
            rep["candidate_concept_ids"]=sorted({h["concept_id"] for h in group})
            rep["concept_id"]="CONFLICT::" + "|".join(sorted(families))
            rep["jurisdiction_scope"]="AMBIGUOUS"
            rep["orientation_layer"]="ambiguous"
            rep["orientation_include"]=0
            rep["primary_include"]=max(h["primary_include"] for h in group)
            rep["ambiguity"]="exact_span_family_conflict"
            resolved.append(rep)

    return resolved

def select_nonoverlap(hits, blocked=None):
    blocked=blocked or []
    ordered=sorted(hits,key=lambda h:(-h["span_length"],h["start"],h["surface_id"]))
    accepted=[]
    for h in ordered:
        if any(overlaps(h,b) for b in blocked): continue
        if any(overlaps(h,a) for a in accepted): continue
        accepted.append(h)
    return sorted(accepted,key=lambda h:(h["start"],h["end"]))

def match_text(text,surface_forms_csv,concepts_csv,language=None):
    lex=load_rows(surface_forms_csv)
    concepts=load_rows(concepts_csv)
    meta={r["concept_id"]:r for r in concepts}
    if language:
        lex=[r for r in lex if r["language"]==language]

    raw_hits=generate_hits(text,lex,meta)
    deduped_hits=resolve_exact_span_collisions(raw_hits)

    by=defaultdict(list)
    for h in deduped_hits:
        by[h["category"]].append(h)

    accepted={}
    accepted["remedy"]=select_nonoverlap(by["remedy"])
    accepted["doctrine"]=select_nonoverlap(by["doctrine"],accepted["remedy"])
    accepted["procedure"]=select_nonoverlap(by["procedure"])
    accepted["party_arg"]=select_nonoverlap(by["party_arg"])

    hits=[]
    for cat in CATEGORY_ORDER:
        hits.extend(accepted[cat])
    return sorted(hits,key=lambda h:(h["start"],h["end"],h["category"]))

def summarize(hits):
    s={}
    for cat in CATEGORY_ORDER:
        hs=[h for h in hits if h["category"]==cat and h["primary_include"]]
        s[cat]={
            "count":len(hs),
            "unique_concept_count":len({h["concept_id"] for h in hs}),
            "orientation_hit_count":sum(h["orientation_include"] for h in hs),
        }
    return s

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("text_file")
    p.add_argument("--surface-forms",default="surface_forms.csv")
    p.add_argument("--concepts",default="concepts.csv")
    p.add_argument("--language",choices=["ko","en"],default=None)
    p.add_argument("--out",default="hits.jsonl")
    a=p.parse_args()

    text=Path(a.text_file).read_text(encoding="utf-8")
    hits=match_text(text,a.surface_forms,a.concepts,a.language)

    with open(a.out,"w",encoding="utf-8") as f:
        for h in hits:
            f.write(json.dumps(h,ensure_ascii=False)+"\n")
    print(json.dumps(summarize(hits),ensure_ascii=False,indent=2))
