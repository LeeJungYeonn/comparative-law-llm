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
        return re.compile(row["surface_form"],re.IGNORECASE|re.DOTALL)
    pat=re.escape(row["surface_form"]).replace(r"\ ",r"\s+")
    return re.compile(pat,re.IGNORECASE)

def generate_hits(text, lex_rows, concept_meta):
    out=[]
    for row in lex_rows:
        c=concept_meta[row["concept_id"]]
        for m in compile_pattern(row).finditer(text):
            out.append({
                "surface_id":row["surface_id"],"concept_id":row["concept_id"],"category":c["category"],
                "jurisdiction_scope":c["jurisdiction_scope"],"orientation_layer":c["orientation_layer"],
                "language":row["language"],"matched_text":text[m.start():m.end()],
                "start":m.start(),"end":m.end(),"span_length":m.end()-m.start(),
                "primary_include":int(row["primary_include"]),"orientation_include":int(row["orientation_include"]),
                "ambiguity":row["ambiguity"]
            })
    return out

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
    if language: lex=[r for r in lex if r["language"]==language]
    by=defaultdict(list)
    for h in generate_hits(text,lex,meta): by[h["category"]].append(h)
    accepted={}
    accepted["remedy"]=select_nonoverlap(by["remedy"])
    accepted["doctrine"]=select_nonoverlap(by["doctrine"],accepted["remedy"])
    accepted["procedure"]=select_nonoverlap(by["procedure"])
    accepted["party_arg"]=select_nonoverlap(by["party_arg"])
    hits=[]
    for cat in CATEGORY_ORDER: hits.extend(accepted[cat])
    return sorted(hits,key=lambda h:(h["start"],h["end"],h["category"]))

def summarize(hits):
    s={}
    for cat in CATEGORY_ORDER:
        hs=[h for h in hits if h["category"]==cat and h["primary_include"]]
        s[cat]={"count":len(hs),"unique_concept_count":len({h["concept_id"] for h in hs}),
                "orientation_hit_count":sum(h["orientation_include"] for h in hs)}
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
        for h in hits: f.write(json.dumps(h,ensure_ascii=False)+"\n")
    print(json.dumps(summarize(hits),ensure_ascii=False,indent=2))
