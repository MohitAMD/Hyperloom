#!/usr/bin/env python3
"""Long-context needle retrieval, giving the reasoning model room to finish."""
import json, urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "/shared_inference/models_blog/GLM-5.1-FP8"


def chat(prompt, max_tokens=2048):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]


def make_haystack(n_filler, needle, frac):
    filler = ("The garden was quiet in the afternoon sun, and the old cat "
              "slept peacefully on the warm stone wall near the fountain. ")
    units = [filler] * n_filler
    units.insert(int(len(units) * frac), " " + needle + " ")
    return "".join(units)


cases = [
    (1000, "The secret vault code for Buenos Aires is 47823.", 0.25,
     "What is the secret vault code for Buenos Aires?", "47823"),
    (1400, "Dr. Halvorsen hid the master key inside locker number 6391 at the north station.", 0.60,
     "Which locker number did Dr. Halvorsen hide the master key in?", "6391"),
    (1600, "The rare blue orchid blooms only in the village of Tavistock every eleventh year.", 0.85,
     "In which village does the rare blue orchid bloom?", "Tavistock"),
]

ok_n = 0
for n, needle, frac, q, gold in cases:
    ctx = make_haystack(n, needle, frac)
    approx_tok = len(ctx) // 4
    prompt = ("Read the following long document carefully, then answer the "
              "question at the end.\n\n<document>\n" + ctx +
              "\n</document>\n\nQuestion: " + q +
              "\nGive your final answer clearly.")
    try:
        out = chat(prompt)
    except Exception as e:
        out = f"<ERROR {e}>"
    ok = gold.lower() in out.lower()
    ok_n += ok
    print(f"[{'PASS' if ok else 'FAIL'}] ~{approx_tok} ctx tok, gold={gold!r}")
    print("   tail:", out.strip().replace("\n", " ")[-200:])
print(f"=== NEEDLE (long-ctx, full reasoning): {ok_n}/{len(cases)} ===")
