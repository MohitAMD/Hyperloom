#!/usr/bin/env python3
"""Accuracy gate for the patched DSA sparse-MLA indexer kernel.
Runs long-context needle-in-haystack retrieval (~16-32k tokens) and a small
gsm8k-style math subset against the local vLLM server, checking against
ground-truth answers. The concern is the e4m3fn->fnuz bitcast trick corrupting
numerics; if it did, long-context retrieval and multi-step math would fail.
"""
import json, re, sys, urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "/shared_inference/models_blog/GLM-5.1-FP8"


def chat(prompt, max_tokens=256, temperature=0.0):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]


def make_haystack(n_filler, needle, needle_pos_frac):
    filler = ("The garden was quiet in the afternoon sun, and the old cat "
              "slept peacefully on the warm stone wall near the fountain. ")
    units = [filler] * n_filler
    pos = int(len(units) * needle_pos_frac)
    units.insert(pos, " " + needle + " ")
    return "".join(units)


def run_needles():
    # ~16-32k tokens: filler unit ~20 tokens, so 1200 units ~= 24k tokens.
    cases = [
        (1000, "The secret vault code for Buenos Aires is 47823.",
         0.25, "What is the secret vault code for Buenos Aires? Answer with just the number.", "47823"),
        (1400, "Dr. Halvorsen hid the master key inside locker number 6391 at the north station.",
         0.60, "Which locker number did Dr. Halvorsen hide the master key in? Answer with just the number.", "6391"),
        (1600, "The rare blue orchid blooms only in the village of Tavistock every eleventh year.",
         0.85, "In which village does the rare blue orchid bloom? Answer with just the village name.", "Tavistock"),
    ]
    results = []
    for n, needle, frac, question, gold in cases:
        ctx = make_haystack(n, needle, frac)
        prompt = ("Read the following long document carefully, then answer the "
                  "question at the end.\n\n<document>\n" + ctx +
                  "\n</document>\n\nQuestion: " + question)
        try:
            out = chat(prompt, max_tokens=64)
        except Exception as e:
            out = f"<ERROR {e}>"
        ok = gold.lower() in out.lower()
        results.append((gold, ok, out.strip().replace("\n", " ")[:120]))
    return results


GSM = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?", 72),
    ("Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. How much did she earn?", 10),
    ("Betty is saving for a $100 wallet. She has half of the money she needs. Her parents give her $15 and her grandparents twice as much as her parents. How much more money does Betty need?", 5),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", 624),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", 3),
    ("Every day, Wendi feeds each of her chickens 3 cups of feed. She has 20 chickens. How many cups of feed does she need per day?", 60),
    ("Kylar wants to buy 16 glasses. Each glass costs $5, but every second glass costs 60% of the price. How much does he need to pay?", 64),
    ("Toula went to the bakery and bought 3 dozen donuts for $68 per dozen. How much did she pay in total?", 204),
    ("Carla downloads a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through it restarts. How many minutes total does it take? (assume it re-downloads from scratch)", 160),
    ("John buys 3 shirts. The first costs $20, and each subsequent shirt costs $5 more than the previous. How much did he spend in total?", 75),
    ("A cup of flour makes 12 cookies. How many cups are needed to make 96 cookies?", 8),
    ("Tim has 30 apples. He gives 12 to his sister and buys 7 more. How many apples does he have now?", 25),
]


def extract_num(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def run_gsm():
    results = []
    for q, gold in GSM:
        prompt = q + "\nThink briefly, then give the final numeric answer on the last line as 'Answer: <number>'."
        try:
            out = chat(prompt, max_tokens=512)
        except Exception as e:
            out = f"<ERROR {e}>"
        m = re.search(r"Answer:\s*\$?(-?[\d,]+\.?\d*)", out)
        pred = (m.group(1).replace(",", "") if m else extract_num(out))
        ok = False
        try:
            ok = pred is not None and abs(float(pred) - gold) < 1e-6
        except Exception:
            ok = False
        results.append((gold, pred, ok, out.strip().replace("\n", " ")[-100:]))
    return results


if __name__ == "__main__":
    print("=== NEEDLE-IN-HAYSTACK (long-context retrieval) ===")
    nres = run_needles()
    n_ok = sum(1 for _, ok, _ in nres if ok)
    for gold, ok, out in nres:
        print(f"  [{'PASS' if ok else 'FAIL'}] gold={gold!r} -> {out!r}")
    print(f"  needle score: {n_ok}/{len(nres)}")
    print("=== GSM8K SUBSET (multi-step math) ===")
    gres = run_gsm()
    g_ok = sum(1 for *_, ok, _ in [(g, p, ok, o) for g, p, ok, o in gres] if ok)
    for gold, pred, ok, tail in gres:
        print(f"  [{'PASS' if ok else 'FAIL'}] gold={gold} pred={pred}")
    print(f"  gsm score: {g_ok}/{len(gres)}")
    print(f"=== TOTAL: needle {n_ok}/{len(nres)}, gsm {g_ok}/{len(gres)} ===")
