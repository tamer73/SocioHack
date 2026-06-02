import os
import json
import time
import sys
from typing import Tuple, Dict, List
import concurrent.futures
import hashlib
import re

# Add repo root to path for src.gemini_client
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_REPO_ROOT)
from src.gemini_client import call_llm_gemini

# --- Configuration ---
# MINED_DIR is where the RL training artifacts (loopholes_<scenario>_fictional.json)
# live. Training writes them to the directory it is launched from (the repo root by
# default); collect them into rl_results_fictional/ or point SOCIOHACK_MINED_DIR
# at wherever they are.
GT_DIR = os.path.join(_REPO_ROOT, "data/fictional")
MINED_DIR = os.environ.get("SOCIOHACK_MINED_DIR", os.path.join(_REPO_ROOT, "rl_results_fictional"))

GEMINI_MODEL_NAME = "gemini-3-flash-preview"
OUTPUT_RESULTS_FILE = os.path.join(MINED_DIR, "eval_fictional_rl_results.json")
PAIR_CACHE_FILE = os.path.join(MINED_DIR, "eval_fictional_rl_cache.json")
FINAL_MARKDOWN = os.path.join(MINED_DIR, "eval_fictional_rl_summary.md")
FINAL_CSV = os.path.join(MINED_DIR, "eval_fictional_rl_summary.csv")

# Concurrency settings
MAX_WORKERS = 10

# Top-K settings
TOP_KS = [1, 3, 5, 10, 'full']


def call_gemini_api_wrapped(prompt_text: str, temperature: float = 0.0) -> Tuple[str, bool]:
    try:
        resp = call_llm_gemini(
            prompt=prompt_text,
            model_name=GEMINI_MODEL_NAME,
            temperature=temperature,
            max_tokens=8192,
            thinking_level="minimal"
        )
        if resp:
            return resp, True
        return "", False
    except Exception as e:
        print(f"Error calling Gemini via client: {e}")
        return "", False


def get_hash(text1: str, text2: str) -> str:
    combined = (text1.strip() + "|||" + text2.strip()).encode('utf-8')
    return hashlib.md5(combined).hexdigest()


def check_loophole_matches_mined_list(loophole_idx: int, loophole_text: str, mined_list: List[str], dataset_name: str, cache_store: dict) -> List[Dict]:
    """Compare a known loophole against mined strategies. Match = mined strategy exploits this loophole."""
    results = []
    needed_mined_indices = []

    for m_idx, m_text in enumerate(mined_list):
        pair_hash = get_hash(m_text, loophole_text)
        res = {
            "mined_index": m_idx,
            "mined_text": m_text,
            "loophole_index": loophole_idx,
            "loophole_text": loophole_text,
            "match": False,
            "reason": "Initial state"
        }
        if pair_hash in cache_store:
            cached_res = cache_store[pair_hash]
            res["match"] = cached_res.get("match", False)
            res["reason"] = cached_res.get("reason", "cached")
            results.append(res)
        else:
            needed_mined_indices.append(m_idx)
            results.append(res)

    if not needed_mined_indices:
        return results

    mined_descriptions = "\n".join([f"Mined Strategy {idx}: \"{mined_list[idx]}\"" for idx in needed_mined_indices])

    prompt = f"""
You are an expert legal analyst and logic judge.
Task: Compare the "Known Loophole" (a documented vulnerability) against a list of "Mined Strategies" for Dataset "{dataset_name}".

Known Loophole (the vulnerability):
"{loophole_text}"

Mined Strategies (model-discovered exploitation strategies):
{mined_descriptions}

For each Mined Strategy listed above, determine if it exploits or targets this exact loophole/vulnerability.
A "match" exists if the mined strategy describes exploiting the same underlying vulnerability as the known loophole.

Output the results strictly in the following JSON format:
{{
  "evaluations": [
    {{
      "mined_index": <index corresponding to Mined Strategy index>,
      "match": <true or false>,
      "reason": "<short explanation>"
    }},
    ...
  ]
}}
"""
    max_parse_retries = 5
    last_err = ""

    for attempt in range(max_parse_retries):
        temp = 0.1 * attempt
        response_text, success = call_gemini_api_wrapped(prompt, temperature=temp)

        if not success:
            for idx in needed_mined_indices:
                results[idx]["reason"] = "API call failed"
            return results

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1:
                    data = json.loads(response_text[start:end+1])
                else:
                    raise ValueError("No JSON found")

            evals = data.get("evaluations", [])
            found_indices = set()
            for ev in evals:
                m_idx = ev.get("mined_index")
                if m_idx is not None and m_idx in needed_mined_indices:
                    is_match = ev.get("match", False)
                    reason = ev.get("reason", "")
                    results[m_idx]["match"] = is_match
                    results[m_idx]["reason"] = reason
                    pair_hash = get_hash(mined_list[m_idx], loophole_text)
                    cache_store[pair_hash] = {"match": is_match, "reason": reason}
                    found_indices.add(m_idx)

            if all(idx in found_indices for idx in needed_mined_indices):
                return results
            else:
                last_err = f"Missing indices in response: {set(needed_mined_indices) - found_indices}"

        except Exception as e:
            last_err = f"Parse error: {str(e)}. Response: {response_text}"

    for idx in needed_mined_indices:
        if results[idx]["reason"] == "Initial state":
            results[idx]["reason"] = f"Parse failed: {last_err}"
    return results


def load_json(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_json(filepath, data):
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    all_results = load_json(OUTPUT_RESULTS_FILE)
    pair_cache = load_json(PAIR_CACHE_FILE)
    print(f"Loaded existing results. Cache size: {len(pair_cache)}")

    final_metrics_summary = []

    print(f"\n==================================================")
    print(f"========== EVALUATING FICTIONAL (RL) ==========")
    print(f"==================================================")

    if not os.path.exists(GT_DIR):
        print(f"Error: GT Dataset directory not found: {GT_DIR}")
        return

    if not os.path.exists(MINED_DIR):
        print(f"Error: Mined directory not found: {MINED_DIR}")
        return

    dataset_files = [f for f in os.listdir(GT_DIR) if f.endswith(".json")]
    dataset_files.sort()

    method = "RL"
    if method not in all_results:
        all_results[method] = {}

    method_mined_lists = {}
    loophole_lists = {}

    for dataset_file in dataset_files:
        dataset_name = dataset_file.replace(".json", "")

        gt_path = os.path.join(GT_DIR, dataset_file)
        loophole_list = []
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                loophole_list = gt_data.get("loopholes", [])
                if not isinstance(loophole_list, list):
                    loophole_list = [loophole_list] if loophole_list else []
                loophole_list = [
                    item if isinstance(item, str) else item.get("text", str(item))
                    for item in loophole_list
                ]
        except Exception as e:
            print(f"Error reading GT {dataset_file}: {e}")
            continue
        loophole_lists[dataset_name] = loophole_list

        # Training writes: loopholes_{dataset_name}_fictional.json
        mined_path = os.path.join(MINED_DIR, f"loopholes_{dataset_name}_fictional.json")
        mined_list = []
        if os.path.exists(mined_path):
            try:
                with open(mined_path, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                    if "loopholes" in m_data:
                        records = m_data.get("loopholes", [])
                    else:
                        records = m_data.get("top_records", [])
                        records = sorted(records, key=lambda x: x.get("rank", 999))
                    for rec in records:
                        summary = rec.get("strategy_summary", "")
                        if summary:
                            mined_list.append(summary)
            except Exception as e:
                print(f"Error reading Mined {mined_path}: {e}")
        else:
            print(f"      [Skip] Mined data not found at {mined_path}")

        method_mined_lists[dataset_name] = mined_list

        needs_eval = True
        if dataset_name in all_results[method] and all_results[method][dataset_name] is not None:
            has_error = any("API call failed" in str(r.get("reason", "")) or "Parse failed" in str(r.get("reason", "")) for r in all_results[method][dataset_name])
            if not has_error:
                if len(all_results[method][dataset_name]) == len(mined_list) * len(loophole_list):
                    needs_eval = False

        if not needs_eval:
            continue

        print(f"  Processing {dataset_name} for API Judge... (Loopholes: {len(loophole_list)}, Mined: {len(mined_list)})")

        if not loophole_list or not mined_list:
            all_results[method][dataset_name] = []
            save_json(OUTPUT_RESULTS_FILE, all_results)
            continue

        loophole_futures = []
        dataset_matrix = [None] * len(loophole_list)

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for l_i, l_text in enumerate(loophole_list):
                loophole_futures.append(
                    executor.submit(check_loophole_matches_mined_list, l_i, l_text, mined_list, dataset_name, pair_cache)
                )

            done_count = 0
            total_loophole = len(loophole_futures)

            for future in concurrent.futures.as_completed(loophole_futures):
                res_list = future.result()
                if res_list:
                    l_idx = res_list[0]["loophole_index"]
                    dataset_matrix[l_idx] = res_list
                done_count += 1
                if done_count % 1 == 0 or done_count == total_loophole:
                    print(f"      ... evaluated {done_count}/{total_loophole} loopholes", end="\r")
            print("")

        dataset_results = []
        for m_row in dataset_matrix:
            if m_row:
                dataset_results.extend(m_row)

        all_results[method][dataset_name] = dataset_results
        save_json(OUTPUT_RESULTS_FILE, all_results)
        save_json(PAIR_CACHE_FILE, pair_cache)

    print(f"\n  --- Calculating Metrics for subset Top-Ks ({method}) ---")
    for k in TOP_KS:
        setting_metrics = []
        for dataset_name in loophole_lists.keys():
            loophole_list = loophole_lists[dataset_name]
            full_mined_list = method_mined_lists[dataset_name]

            if k == 'full':
                mined_list = full_mined_list
            else:
                mined_list = full_mined_list[:k]

            if not loophole_list and not mined_list:
                setting_metrics.append({"dataset": dataset_name, "p": 0.0, "r": 0.0, "f1": 0.0, "mined": 0, "gt": 0})
                continue
            elif not loophole_list:
                setting_metrics.append({"dataset": dataset_name, "p": 0.0, "r": 0.0, "f1": 0.0, "mined": len(mined_list), "gt": 0})
                continue
            elif not mined_list:
                setting_metrics.append({"dataset": dataset_name, "p": 0.0, "r": 0.0, "f1": 0.0, "mined": 0, "gt": len(loophole_list)})
                continue

            dataset_results = all_results[method].get(dataset_name, [])
            matches = [r for r in dataset_results if r.get("match", False) and r.get("mined_index", 999) < len(mined_list)]

            tp_gt = len(set(m["loophole_index"] for m in matches))
            tp_mined = len(set(m["mined_index"] for m in matches))

            recall = tp_gt / len(loophole_list) if len(loophole_list) > 0 else 0.0
            precision = tp_mined / len(mined_list) if len(mined_list) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            setting_metrics.append({
                "dataset": dataset_name, "p": precision, "r": recall, "f1": f1,
                "mined": len(mined_list), "gt": len(loophole_list)
            })

        if setting_metrics:
            avg_p = sum(d["p"] for d in setting_metrics) / len(setting_metrics)
            avg_r = sum(d["r"] for d in setting_metrics) / len(setting_metrics)
            avg_f1 = sum(d["f1"] for d in setting_metrics) / len(setting_metrics)
            print(f"  [Summary] {method} {k if k == 'full' else 'top_'+str(k)} | P: {avg_p:.4f} | R: {avg_r:.4f} | F1: {avg_f1:.4f}")

            final_metrics_summary.append({
                "method": method,
                "top_k": k if k == 'full' else f"top_{k}",
                "p": avg_p,
                "r": avg_r,
                "f1": avg_f1
            })

    with open(FINAL_MARKDOWN, "w", encoding="utf-8") as f_md, open(FINAL_CSV, "w", encoding="utf-8") as f_csv:
        f_md.write("# Fictional Evaluation Results (RL, Compare vs Loopholes)\n\n")
        f_md.write("| Method | Top-K | Precision | Recall | F1 Score |\n")
        f_md.write("|---|---|---|---|---|\n")
        f_csv.write("Method,Top-K,Precision,Recall,F1_Score\n")

        def sort_key(item):
            tk = item['top_k']
            tk_rank = 999 if tk == 'full' else int(tk.split('_')[1])
            return (item['method'], tk_rank)

        final_metrics_summary.sort(key=sort_key)
        for record in final_metrics_summary:
            f_md.write(f"| {record['method']} | {record['top_k']} | {record['p']:.4f} | {record['r']:.4f} | {record['f1']:.4f} |\n")
            f_csv.write(f"{record['method']},{record['top_k']},{record['p']:.4f},{record['r']:.4f},{record['f1']:.4f}\n")


if __name__ == "__main__":
    main()
