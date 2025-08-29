import argparse, os, json
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env if present
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

REQUIRED_COLS = [
    "respondent_id", "version", "screen",
    "item_1", "item_2", "item_3", "item_4",
    "best_item", "worst_item", "status"
]

# ---------- helpers ----------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {}
    for i in range(1,5):
        for cnd in [f"item_{i}", f"item{i}", f"item {i}"]:
            if cnd in df.columns:
                rename_map[cnd] = f"item_{i}"
                break
    if "best" in df.columns and "best_item" not in df.columns:
        rename_map["best"] = "best_item"
    if "worst" in df.columns and "worst_item" not in df.columns:
        rename_map["worst"] = "worst_item"
    return df.rename(columns=rename_map)

def _validate_columns(df: pd.DataFrame) -> None:
    if "status" not in df.columns:
        df["status"] = np.where(df["best_item"].notna() & df["worst_item"].notna(), "observed", "missing")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}. Expected: {REQUIRED_COLS}")

def _as_str(x):
    return None if pd.isna(x) else str(x)

def _build_item_scores(obs_rows: pd.DataFrame):
    scores, best_counts, worst_counts = {}, {}, {}
    for _, r in obs_rows.iterrows():
        items = [_as_str(r["item_1"]), _as_str(r["item_2"]), _as_str(r["item_3"]), _as_str(r["item_4"])]
        for it in items:
            if it is None: continue
            scores.setdefault(it, 0); best_counts.setdefault(it, 0); worst_counts.setdefault(it, 0)
        b = _as_str(r["best_item"]); w = _as_str(r["worst_item"])
        if b is not None: scores[b] = scores.get(b,0)+1; best_counts[b] = best_counts.get(b,0)+1
        if w is not None: scores[w] = scores.get(w,0)-1; worst_counts[w] = worst_counts.get(w,0)+1
    return scores, best_counts, worst_counts

def _build_global_scores(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Global net scores per version (Best - Worst) from ALL observed rows."""
    glob: Dict[str, Dict[str,int]] = {}
    obs = df[df["status"].astype(str).str.lower().eq("observed")]
    for ver, g in obs.groupby("version", dropna=False, sort=False):
        scores: Dict[str,int] = {}
        for _, r in g.iterrows():
            items = [_as_str(r["item_1"]), _as_str(r["item_2"]), _as_str(r["item_3"]), _as_str(r["item_4"])]
            for it in items:
                if it is None: continue
                scores.setdefault(it, 0)
            b = _as_str(r["best_item"]); w = _as_str(r["worst_item"])
            if b is not None: scores[b] = scores.get(b,0)+1
            if w is not None: scores[w] = scores.get(w,0)-1
        glob[str(ver)] = scores
    return glob

def _choose_pair_for_screen(screen_items: List[str], base_scores: Dict[str,int], seen_best:set, seen_worst:set):
    scores = dict(base_scores)  # do not mutate caller
    for it in screen_items:
        scores.setdefault(it, 0)

    best_rank = sorted(screen_items, key=lambda x: (-scores.get(x,0), str(x)))
    worst_rank = sorted(screen_items, key=lambda x: (scores.get(x,0), str(x)))

    def try_pick():
        for b in best_rank:
            if b in seen_worst: 
                continue
            for w in worst_rank:
                if b == w: continue
                if w in seen_best: 
                    continue
                return b, w
        for b in best_rank:
            for w in worst_rank:
                if b != w: return b, w
        return best_rank[0], worst_rank[-1] if worst_rank[-1]!=best_rank[0] else worst_rank[0]

    b,w = try_pick()

    def conf_best():
        if len(best_rank)==1: return 1.0
        s0 = scores.get(best_rank[0],0); s1 = scores.get(best_rank[1],0)
        rng = max(1, max(scores.get(x,0) for x in screen_items) - min(scores.get(x,0) for x in screen_items))
        return max(0.5, min(1.0, 0.5 + (s0-s1)/(2*rng)))
    def conf_worst():
        if len(worst_rank)==1: return 1.0
        s0 = scores.get(worst_rank[0],0); s1 = scores.get(worst_rank[1],0)
        rng = max(1, max(scores.get(x,0) for x in screen_items) - min(scores.get(x,0) for x in screen_items))
        return max(0.5, min(1.0, 0.5 + (s1-s0)/(2*rng)))
    return b,w,float(round(conf_best(),3)),float(round(conf_worst(),3))

# ---------- heuristic: individual ----------
def augment_maxdiff(input_path: str, output_path: str, sheet_name=0, method_label="heuristic-maxdiff-v1"):
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    df = _normalize_cols(df); _validate_columns(df)

    if "imputed_flag" not in df.columns: df["imputed_flag"] = 0
    if "confidence_best" not in df.columns: df["confidence_best"] = np.nan
    if "confidence_worst" not in df.columns: df["confidence_worst"] = np.nan
    df["method"] = method_label

    sort_cols = [c for c in ["respondent_id","version","screen"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    group_cols = [c for c in ["respondent_id","version"] if c in df.columns]

    for keys, g in df.groupby(group_cols, dropna=False, sort=False):
        obs_mask = g["status"].astype(str).str.lower().eq("observed")
        obs_rows = g[obs_mask]
        scores, _, _ = _build_item_scores(obs_rows)

        seen_best = set([str(x) for x in obs_rows["best_item"].dropna().astype(str)])
        seen_worst = set([str(x) for x in obs_rows["worst_item"].dropna().astype(str)])

        g_sorted = g.sort_values(["screen"]).copy()
        for _, row in g_sorted.iterrows():
            idx = int(row.name)
            if str(row["status"]).lower() == "observed":
                if pd.isna(row.get("confidence_best", np.nan)): df.at[idx,"confidence_best"]=1.0
                if pd.isna(row.get("confidence_worst", np.nan)): df.at[idx,"confidence_worst"]=1.0
                b = _as_str(row["best_item"]); w = _as_str(row["worst_item"])
                if b is not None: scores[b]=scores.get(b,0)+1; seen_best.add(b)
                if w is not None: scores[w]=scores.get(w,0)-1; seen_worst.add(w)
                continue
            items = [str(row["item_1"]),str(row["item_2"]),str(row["item_3"]),str(row["item_4"])]
            items = [i for i in items if i and i.lower()!="nan"]
            best,worst,cb,cw = _choose_pair_for_screen(items, scores, seen_best, seen_worst)
            df.at[idx,"best_item"]=best; df.at[idx,"worst_item"]=worst
            df.at[idx,"imputed_flag"]=1; df.at[idx,"confidence_best"]=cb; df.at[idx,"confidence_worst"]=cw
            scores[best]=scores.get(best,0)+1; scores[worst]=scores.get(worst,0)-1
            seen_best.add(best); seen_worst.add(worst)

    col_order = ["respondent_id","version","screen","item_1","item_2","item_3","item_4","best_item","worst_item","status","imputed_flag","confidence_best","confidence_worst","method"]
    final_cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    df = df[final_cols]
    if output_path: df.to_excel(output_path, index=False)
    return df

# ---------- heuristic: global ----------
def augment_maxdiff_global(input_path: str, output_path: str, sheet_name=0, method_label="heuristic-global-maxdiff-v1"):
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    df = _normalize_cols(df); _validate_columns(df)

    if "imputed_flag" not in df.columns: df["imputed_flag"] = 0
    if "confidence_best" not in df.columns: df["confidence_best"] = np.nan
    if "confidence_worst" not in df.columns: df["confidence_worst"] = np.nan
    df["method"] = method_label

    sort_cols = [c for c in ["respondent_id","version","screen"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    group_cols = [c for c in ["respondent_id","version"] if c in df.columns]

    global_scores_by_version = _build_global_scores(df)

    for (_, ver), g in df.groupby(group_cols, dropna=False, sort=False):
        seen_best = set([str(x) for x in g[g["status"].astype(str).str.lower().eq("observed")]["best_item"].dropna().astype(str)])
        seen_worst = set([str(x) for x in g[g["status"].astype(str).str.lower().eq("observed")]["worst_item"].dropna().astype(str)])

        base_scores = global_scores_by_version.get(str(ver), {})
        g_sorted = g.sort_values(["screen"]).copy()
        for _, row in g_sorted.iterrows():
            idx = int(row.name)
            if str(row["status"]).lower() == "observed":
                if pd.isna(row.get("confidence_best", np.nan)): df.at[idx,"confidence_best"]=1.0
                if pd.isna(row.get("confidence_worst", np.nan)): df.at[idx,"confidence_worst"]=1.0
                b = _as_str(row["best_item"]); w = _as_str(row["worst_item"])
                if b is not None: seen_best.add(b)
                if w is not None: seen_worst.add(w)
                continue
            items = [str(row["item_1"]),str(row["item_2"]),str(row["item_3"]),str(row["item_4"])]
            items = [i for i in items if i and i.lower()!="nan"]
            best,worst,cb,cw = _choose_pair_for_screen(items, base_scores, seen_best, seen_worst)
            df.at[idx,"best_item"]=best; df.at[idx,"worst_item"]=worst
            df.at[idx,"imputed_flag"]=1; df.at[idx,"confidence_best"]=cb; df.at[idx,"confidence_worst"]=cw
            seen_best.add(best); seen_worst.add(worst)

    col_order = ["respondent_id","version","screen","item_1","item_2","item_3","item_4","best_item","worst_item","status","imputed_flag","confidence_best","confidence_worst","method"]
    final_cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    df = df[final_cols]
    if output_path: df.to_excel(output_path, index=False)
    return df

# ---------- LLM helpers ----------
def _format_obs_summary(obs_rows: pd.DataFrame) -> str:
    best_counts = {}; worst_counts = {}
    for _, r in obs_rows.iterrows():
        items = [r.get("item_1"),r.get("item_2"),r.get("item_3"),r.get("item_4")]
        b = r.get("best_item"); w = r.get("worst_item")
        if pd.notna(b): best_counts[str(b)] = best_counts.get(str(b), 0) + 1
        if pd.notna(w): worst_counts[str(w)] = worst_counts.get(str(w), 0) + 1
        for it in items:
            if pd.isna(it): continue
            it = str(it); best_counts.setdefault(it, 0); worst_counts.setdefault(it, 0)
    rows = []
    for it in sorted(set(list(best_counts.keys()) + list(worst_counts.keys()))):
        b = best_counts.get(it,0); w = worst_counts.get(it,0)
        rows.append({"item": it, "best": int(b), "worst": int(w), "net": int(b-w)})
    return json.dumps(rows, ensure_ascii=False)

def _format_global_summary(df: pd.DataFrame, version_val) -> str:
    obs = df[df["status"].astype(str).str.lower().eq("observed")]
    if "version" in df.columns:
        obs = obs[obs["version"].astype(str) == str(version_val)]
    best_counts = {}; worst_counts = {}
    for _, r in obs.iterrows():
        items = [r.get("item_1"),r.get("item_2"),r.get("item_3"),r.get("item_4")]
        b = r.get("best_item"); w = r.get("worst_item")
        if pd.notna(b): best_counts[str(b)] = best_counts.get(str(b), 0) + 1
        if pd.notna(w): worst_counts[str(w)] = worst_counts.get(str(w), 0) + 1
        for it in items:
            if pd.isna(it): continue
            it = str(it); best_counts.setdefault(it, 0); worst_counts.setdefault(it, 0)
    rows = []
    for it in sorted(set(list(best_counts.keys()) + list(worst_counts.keys()))):
        b = best_counts.get(it,0); w = worst_counts.get(it,0)
        rows.append({"item": it, "best": int(b), "worst": int(w), "net": int(b-w)})
    return json.dumps(rows, ensure_ascii=False)

# ---------- LLM: individual ----------
def augment_maxdiff_llm(input_path: str, output_path: str, sheet_name=0, model="gpt-4o-mini", method_label="llm-predictive-maxdiff-v1", api_key_env="OPENAI_API_KEY", temperature=0.0, max_retries=2):
    try:
        from openai import OpenAI
    except Exception:
        OpenAI = None

    df = pd.read_excel(input_path, sheet_name=sheet_name)
    df = _normalize_cols(df); _validate_columns(df)

    if "imputed_flag" not in df.columns: df["imputed_flag"] = 0
    if "confidence_best" not in df.columns: df["confidence_best"] = np.nan
    if "confidence_worst" not in df.columns: df["confidence_worst"] = np.nan
    df["method"] = df.get("method", method_label)

    api_key = os.environ.get(api_key_env, "")
    if not api_key or OpenAI is None:
        return augment_maxdiff(input_path, output_path, sheet_name=sheet_name, method_label="heuristic-fallback")

    client = OpenAI(api_key=api_key)
    group_cols = [c for c in ["respondent_id","version"] if c in df.columns]
    df = df.sort_values(group_cols + (["screen"] if "screen" in df.columns else [])).reset_index(drop=True)

    import json as _json
    for keys, g in df.groupby(group_cols, dropna=False, sort=False):
        obs = g[g["status"].astype(str).str.lower().eq("observed")]
        miss = g[g["status"].astype(str).str.lower().eq("missing")]
        if miss.empty: continue
        obs_summary = _format_obs_summary(obs)
        for idx, row in miss.iterrows():
            items = [str(row["item_1"]),str(row["item_2"]),str(row["item_3"]),str(row["item_4"])]
            items = [i for i in items if i and i.lower()!="nan"]
            sys_msg = "You are an expert MaxDiff (Best-Worst Scaling) analyst. Respond ONLY with valid JSON."
            user_msg = {
                "mode": "llm-individual",
                "instructions": "Choose exactly one 'best' and one 'worst' from the four items for this screen.",
                "constraints": [
                    "Best and Worst must be different.",
                    "Both choices must be from 'items_on_screen'.",
                    "Respect observed tendencies in 'observed_summary'.",
                    "Avoid cross-screen contradictions when possible."
                ],
                "observed_summary": _json.loads(obs_summary),
                "items_on_screen": items,
                "respondent_id": str(row.get("respondent_id","")),
                "version": str(row.get("version","")),
                "screen": str(row.get("screen","")),
                "output_schema": {"best_item":"string","worst_item":"string","confidence_best":"0-1 float","confidence_worst":"0-1 float"}
            }
            prompt = _json.dumps(user_msg, ensure_ascii=False)
            content = None
            for _ in range(max_retries+1):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=temperature,
                        messages=[{"role":"system","content":sys_msg},
                                  {"role":"user","content":prompt}],
                        response_format={"type":"json_object"}
                    )
                    content = resp.choices[0].message.content
                    break
                except Exception:
                    content=None
            if content:
                try:
                    data = _json.loads(content)
                    best = str(data["best_item"]); worst = str(data["worst_item"])
                    if best!=worst and best in items and worst in items:
                        df.at[idx,"best_item"]=best; df.at[idx,"worst_item"]=worst
                        df.at[idx,"imputed_flag"]=1
                        df.at[idx,"confidence_best"]=float(data.get("confidence_best",0.8))
                        df.at[idx,"confidence_worst"]=float(data.get("confidence_worst",0.8))
                        df.at[idx,"method"]=method_label
                        continue
                except Exception:
                    pass
            # fallback per-row
            tmp_out = "_tmp_fallback.xlsx"
            heur = augment_maxdiff(input_path, tmp_out, sheet_name=sheet_name, method_label="heuristic-fallback")
            df.at[idx,"best_item"]=heur.loc[idx,"best_item"]; df.at[idx,"worst_item"]=heur.loc[idx,"worst_item"]
            df.at[idx,"imputed_flag"]=1
            df.at[idx,"confidence_best"]=heur.loc[idx,"confidence_best"]
            df.at[idx,"confidence_worst"]=heur.loc[idx,"confidence_worst"]
            df.at[idx,"method"]="heuristic-fallback"

    col_order = ["respondent_id","version","screen","item_1","item_2","item_3","item_4","best_item","worst_item","status","imputed_flag","confidence_best","confidence_worst","method"]
    final_cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    df = df[final_cols]
    if output_path: df.to_excel(output_path, index=False)
    return df

# ---------- LLM: global ----------
def augment_maxdiff_llm_global(input_path: str, output_path: str, sheet_name=0, model="gpt-4o-mini", method_label="llm-global-predictive-maxdiff-v1", api_key_env="OPENAI_API_KEY", temperature=0.0, max_retries=2):
    try:
        from openai import OpenAI
    except Exception:
        OpenAI = None

    df = pd.read_excel(input_path, sheet_name=sheet_name)
    df = _normalize_cols(df); _validate_columns(df)

    if "imputed_flag" not in df.columns: df["imputed_flag"] = 0
    if "confidence_best" not in df.columns: df["confidence_best"] = np.nan
    if "confidence_worst" not in df.columns: df["confidence_worst"] = np.nan
    df["method"] = df.get("method", method_label)

    api_key = os.environ.get(api_key_env, "")
    if not api_key or OpenAI is None:
        return augment_maxdiff_global(input_path, output_path, sheet_name=sheet_name, method_label="heuristic-global-fallback")

    client = OpenAI(api_key=api_key)
    group_cols = [c for c in ["respondent_id","version"] if c in df.columns]
    df = df.sort_values(group_cols + (["screen"] if "screen" in df.columns else [])).reset_index(drop=True)

    import json as _json
    for (rid, ver), g in df.groupby(group_cols, dropna=False, sort=False):
        obs = g[g["status"].astype(str).str.lower().eq("observed")]
        miss = g[g["status"].astype(str).str.lower().eq("missing")]
        if miss.empty: continue
        obs_summary = _format_obs_summary(obs)
        global_summary = _format_global_summary(df, ver)

        for idx, row in miss.iterrows():
            items = [str(row["item_1"]),str(row["item_2"]),str(row["item_3"]),str(row["item_4"])]
            items = [i for i in items if i and i.lower()!="nan"]
            sys_msg = "You are an expert MaxDiff (Best-Worst Scaling) analyst. Respond ONLY with valid JSON."
            user_msg = {
                "mode": "llm-global",
                "instructions": "Choose exactly one 'best' and one 'worst' from the four items for this screen. Prioritize global tendencies; use respondent tendencies as secondary signal.",
                "constraints": [
                    "Best and Worst must be different.",
                    "Both choices must be from 'items_on_screen'.",
                    "Use 'global_observed_summary' as primary prior; use 'respondent_observed_summary' to break ties.",
                    "Avoid cross-screen contradictions when possible for this respondent."
                ],
                "global_observed_summary": _json.loads(global_summary),
                "respondent_observed_summary": _json.loads(obs_summary),
                "items_on_screen": items,
                "respondent_id": str(rid),
                "version": str(ver),
                "screen": str(row.get("screen","")),
                "output_schema": {"best_item":"string","worst_item":"string","confidence_best":"0-1 float","confidence_worst":"0-1 float"}
            }
            prompt = _json.dumps(user_msg, ensure_ascii=False)
            content = None
            for _ in range(max_retries+1):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=temperature,
                        messages=[{"role":"system","content":sys_msg},
                                  {"role":"user","content":prompt}],
                        response_format={"type":"json_object"}
                    )
                    content = resp.choices[0].message.content
                    break
                except Exception:
                    content=None
            if content:
                try:
                    data = _json.loads(content)
                    best = str(data["best_item"]); worst = str(data["worst_item"])
                    if best!=worst and best in items and worst in items:
                        df.at[idx,"best_item"]=best; df.at[idx,"worst_item"]=worst
                        df.at[idx,"imputed_flag"]=1
                        df.at[idx,"confidence_best"]=float(data.get("confidence_best",0.8))
                        df.at[idx,"confidence_worst"]=float(data.get("confidence_worst",0.8))
                        df.at[idx,"method"]=method_label
                        continue
                except Exception:
                    pass
            # fallback per-row: heuristic global
            tmp_out = "_tmp_fallback.xlsx"
            heur = augment_maxdiff_global(input_path, tmp_out, sheet_name=sheet_name, method_label="heuristic-global-fallback")
            df.at[idx,"best_item"]=heur.loc[idx,"best_item"]; df.at[idx,"worst_item"]=heur.loc[idx,"worst_item"]
            df.at[idx,"imputed_flag"]=1
            df.at[idx,"confidence_best"]=heur.loc[idx,"confidence_best"]
            df.at[idx,"confidence_worst"]=heur.loc[idx,"confidence_worst"]
            df.at[idx,"method"]="heuristic-global-fallback"

    col_order = ["respondent_id","version","screen","item_1","item_2","item_3","item_4","best_item","worst_item","status","imputed_flag","confidence_best","confidence_worst","method"]
    final_cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    df = df[final_cols]
    if output_path: df.to_excel(output_path, index=False)
    return df

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sheet", default="0")
    parser.add_argument("--mode", choices=["heuristic-individual","heuristic-global","llm-individual","llm-global"], default="heuristic-individual")
    parser.add_argument("--method-label", default="heuristic-maxdiff-v1")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    args = parser.parse_args()
    sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet

    if args.mode == "heuristic-individual":
        augment_maxdiff(args.input, args.out, sheet_name=sheet, method_label=args.method_label)
    elif args.mode == "heuristic-global":
        augment_maxdiff_global(args.input, args.out, sheet_name=sheet, method_label="heuristic-global-maxdiff-v1")
    elif args.mode == "llm-individual":
        augment_maxdiff_llm(args.input, args.out, sheet_name=sheet, model=args.llm_model, method_label="llm-predictive-maxdiff-v1")
    else:
        augment_maxdiff_llm_global(args.input, args.out, sheet_name=sheet, model=args.llm_model, method_label="llm-global-predictive-maxdiff-v1")

if __name__ == "__main__":
    main()
