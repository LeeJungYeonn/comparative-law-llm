#!/usr/bin/env python3
import csv, re, json
from collections import defaultdict

CATEGORY_ORDER = ["remedy","doctrine","procedure","party_arg"]

def load_rows(path):
    with open(path,encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def overlaps(a,b):
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])

def compile_pattern(row):
    if row["match_type"]=="regex":
        return re.compile(row["surface_form"], re.IGNORECASE)
    pat=re.escape(row["surface_form"]).replace(r"\ ",r"\s+")
    if row["language"]=="en":
        pat=rf"(?<![A-Za-z0-9_]){pat}(?![A-Za-z0-9_])"
    return re.compile(pat,re.IGNORECASE)

def generate_hits(text, rows, meta):
    out=[]
    for row in rows:
        if int(row.get("primary_include","0"))!=1:
            continue
        c=meta[row["concept_id"]]
        for m in compile_pattern(row).finditer(text):
            out.append({
                "surface_id":row["surface_id"],"concept_id":row["concept_id"],
                "concept_family":c["concept_family"],"category":c["category"],
                "jurisdiction_scope":c["jurisdiction_scope"],"orientation_layer":c["orientation_layer"],
                "matched_text":text[m.start():m.end()],"start":m.start(),"end":m.end(),
                "span_length":m.end()-m.start(),"orientation_include":int(row.get("orientation_include","0"))
            })
    return out

def resolve_exact_span_collisions(hits):
    groups=defaultdict(list)
    for h in hits:
        groups[(h["start"],h["end"],h["category"],h["matched_text"].casefold())].append(h)
    resolved=[]
    for _,g in groups.items():
        if len(g)==1:
            x=g[0].copy(); x["candidate_concept_ids"]=[x["concept_id"]]; resolved.append(x); continue
        fams={x["concept_family"] for x in g}; scopes={x["jurisdiction_scope"] for x in g}
        rep=sorted(g,key=lambda x:x["surface_id"])[0].copy()
        rep["candidate_concept_ids"]=sorted({x["concept_id"] for x in g})
        if len(fams)==1 and len(scopes)>1:
            fam=next(iter(fams))
            rep["concept_id"]="SHARED::"+fam
            rep["jurisdiction_scope"]="SHARED"
            rep["orientation_layer"]="shared"
            rep["orientation_include"]=0
        elif len(fams)>1:
            rep["concept_id"]="CONFLICT::"+"|".join(sorted(fams))
            rep["jurisdiction_scope"]="AMBIGUOUS"
            rep["orientation_layer"]="ambiguous"
            rep["orientation_include"]=0
        else:
            rep["orientation_include"]=max(x["orientation_include"] for x in g)
        resolved.append(rep)
    return resolved

def select_nonoverlap(hits,blocked=None):
    blocked=blocked or []
    ordered=sorted(hits,key=lambda h:(-h["span_length"],h["start"],h["surface_id"]))
    acc=[]
    for h in ordered:
        if any(overlaps(h,b) for b in blocked): continue
        if any(overlaps(h,a) for a in acc): continue
        acc.append(h)
    return sorted(acc,key=lambda h:(h["start"],h["end"]))

def match_text(text,surface_csv,concept_csv,language=None):
    rows=load_rows(surface_csv); cs=load_rows(concept_csv); meta={r["concept_id"]:r for r in cs}
    if language: rows=[r for r in rows if r["language"]==language]
    raw=resolve_exact_span_collisions(generate_hits(text,rows,meta))
    by=defaultdict(list)
    for h in raw: by[h["category"]].append(h)
    acc={}
    acc["remedy"]=select_nonoverlap(by["remedy"])
    acc["doctrine"]=select_nonoverlap(by["doctrine"],acc["remedy"])
    acc["procedure"]=select_nonoverlap(by["procedure"])
    acc["party_arg"]=select_nonoverlap(by["party_arg"])
    out=[]
    for cat in CATEGORY_ORDER: out.extend(acc[cat])
    return sorted(out,key=lambda h:(h["start"],h["end"],h["category"]))

def summarize(hits):
    out={}
    for cat in CATEGORY_ORDER:
        hs=[h for h in hits if h["category"]==cat]
        out[cat]={
            "hit_count":len(hs),
            "unique_concept_count":len({h["concept_id"] for h in hs}),
            "unique_concept_family_count":len({h["concept_family"] for h in hs}),
            "orientation_hit_count":sum(h["orientation_include"] for h in hs),
        }
    return out
