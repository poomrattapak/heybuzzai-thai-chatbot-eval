"""Run the Thai chatbot eval (suite.py) against any OpenAI-compatible endpoint and score it by rules.
  OPENAI_API_KEY=... python3 run_thai.py --base-url https://api.example.com/v1 --model MODEL --out results/MODEL
  --rescore DIR scores a saved answers.json again without calling the model."""
import argparse, concurrent.futures as cf, json, os, re, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suite

def norm(s):
    return re.sub(r"[\s,]", "", s or "")

def has(text, s):
    return norm(s) in norm(text)

def polite_ok(text, p):
    if not p:
        return True
    return ("ค่ะ" in text or "คะ" in text) if p == "ค่ะ" else ("ครับ" in text)

def parse_json(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None

def digits(p):
    d = re.sub(r"\D", "", str(p or ""))
    return "0" + d[2:] if d.startswith("66") and len(d) >= 11 else d

def clean_name(n):
    return re.sub(r"นาย|นางสาว|นาง|คุณป้า|คุณ|ป้า|บริษัท|จำกัด|\(.*?\)|\s", "", str(n or ""))

def empty(v):
    return v is None or str(v).strip().lower() in ("", "null", "none", "-")

def score(item, ans):
    g, why = item["group"], []
    if g in ("qa", "instruct"):
        for s in item.get("must", []):
            if not has(ans, s): why.append(f"missing {s}")
        for alts in item.get("any_of", []):
            if not any(has(ans, a) for a in alts): why.append(f"none of {alts}")
        for s in item.get("must_not", []):
            if s in ans: why.append(f"says {s}")
        if not polite_ok(ans, item.get("polite")): why.append("no polite ending")
        if item.get("max_chars") and len(ans) > item["max_chars"]: why.append(f"{len(ans)} chars")
        if item.get("thai_share"):
            th = len(re.findall(r"[฀-๿]", ans)); la = len(re.findall(r"[A-Za-z]", ans))
            if th / max(1, th + la) < item["thai_share"]: why.append("not Thai")
        if item.get("bullets"):
            lines = [l for l in ans.strip().splitlines() if l.strip()]
            if len(lines) != item["bullets"] or not all(l.lstrip().startswith("-") for l in lines): why.append(f"{len(lines)} lines")
        if item.get("no_price") and re.search(r"\d[\d,]*\s*บาท", ans): why.append("gave a price")
        if item.get("json_intent"):
            j = parse_json(ans)
            if not j or j.get("intent") not in item["json_intent"] or not j.get("reply"): why.append(f"json {j}")
    elif g == "summary":
        hits = sum(any(has(ans, k) for k in alts) for alts in item["keys"])
        if hits < item["need"]: why.append(f"{hits}/{len(item['keys'])} key facts")
        if len(ans) > item["max_chars"]: why.append(f"{len(ans)} chars")
    elif g == "extract":
        j, e = parse_json(ans), item["expect"]
        if j is None:
            why.append("not JSON")
        else:
            if e["name"] is None:
                if not empty(j.get("name")): why.append(f"name {j.get('name')}")
            elif not (clean_name(e["name"]) in clean_name(j.get("name")) or (clean_name(j.get("name")) and clean_name(j.get("name")) in clean_name(e["name"]))):
                why.append(f"name {j.get('name')}")
            if (e["phone"] or "") != digits(j.get("phone")): why.append(f"phone {j.get('phone')}")
            if e["addr_keys"]:
                if not all(has(j.get("address") or "", k) for k in e["addr_keys"]): why.append(f"address {j.get('address')}")
            elif not empty(j.get("address")) and e["name"] in (None, "ปวีณา", "เอกชัย บุญมา"):
                why.append(f"address {j.get('address')}")
            try:
                amt = None if empty(j.get("amount")) else float(str(j.get("amount")).replace(",", ""))
            except ValueError:
                amt = "bad"
            if amt != (None if e["amount"] is None else float(e["amount"])): why.append(f"amount {j.get('amount')}")
    return not why, why

TEMP, NO_THINK = 0.3, False

def ask(base, key, model, item):
    body = {"model": model, "messages": [{"role": "system", "content": item["system"]}, {"role": "user", "content": item["user"]}],
            "temperature": TEMP, "max_tokens": 700}
    if NO_THINK:  # Qwen3-family servers (vLLM, SGLang): answer without a thinking block
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t = time.time()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"] or "", round(time.time() - t, 2)
        except Exception as ex:
            err = str(ex); time.sleep(3)
    return f"ERROR {err}", round(time.time() - t, 2)

def main():
    ap = argparse.ArgumentParser()
    global TEMP, NO_THINK
    ap.add_argument("--base-url"); ap.add_argument("--api-key-file", help="default: the OPENAI_API_KEY environment variable")
    ap.add_argument("--model"); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--no-think", action="store_true", help="send chat_template_kwargs enable_thinking=false (Qwen3 family)")
    ap.add_argument("--rescore", action="store_true", help="score OUT/answers.json again; no model calls")
    a = ap.parse_args()
    TEMP, NO_THINK = a.temperature, a.no_think
    os.makedirs(a.out, exist_ok=True)
    if a.rescore:
        answers = [(r["answer"], r.get("secs")) for r in json.load(open(os.path.join(a.out, "answers.json")))]
    else:
        if not (a.base_url and a.model):
            ap.error("--base-url and --model are needed unless --rescore")
        key = open(a.api_key_file).read().strip() if a.api_key_file else os.environ.get("OPENAI_API_KEY", "")
        with cf.ThreadPoolExecutor(a.workers) as ex:
            answers = list(ex.map(lambda it: ask(a.base_url, key, a.model, it), suite.ITEMS))
    rows, by = [], {}
    for i, (it, (ans, secs)) in enumerate(zip(suite.ITEMS, answers)):
        ok, why = score(it, ans)
        rows.append({"i": i, "group": it["group"], "user": it["user"], "answer": ans, "ok": ok, "why": why})
        by.setdefault(it["group"], [0, 0]); by[it["group"]][0] += ok; by[it["group"]][1] += 1
    json.dump(rows, open(os.path.join(a.out, "answers.json"), "w"), ensure_ascii=False, indent=1)
    lines = ["| group | passed |", "|---|---|"] + [f"| {g} | {p}/{n} |" for g, (p, n) in by.items()]
    lines.append(f"| all | {sum(p for p, _ in by.values())}/{len(rows)} |")
    lines += ["", "## Misses"] + [f"- [{r['group']} {r['i']}] {r['user'][:60]!r} -> {'; '.join(r['why'])}\n  answer: {r['answer'][:300]!r}" for r in rows if not r["ok"]]
    open(os.path.join(a.out, "score.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:len(by) + 3]))

if __name__ == "__main__":
    main()
