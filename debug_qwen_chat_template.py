"""
Compare incremental chat-template tokenisation between
Qwen-2.5-7B-Instruct and Qwen-3-4B.

It mimics the exact steps used in Environment.process_chat_format():
1. build prompt → tokenise (add_generation_prompt=True)
2. iterate over completion messages:
     a) rebuild conversation prefix
     b) tokenise (add_generation_prompt=False)
     c) assert the previous tokens are an exact prefix
"""

from transformers import AutoTokenizer

MODELS = {
    "qwen25": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3":  "Qwen/Qwen3-4B"
}

# ---------- toy conversation (same for both) ----------
prompt_msgs = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is 2+2?"}
]

completion_msgs = [
    {"role": "assistant", "content": " 2+2 is 4."},
    {"role": "user",      "content": "Thanks!"},
    {"role": "assistant", "content": " You're welcome."},
]

# ---------- helper ----------
def run_debug(tok_name: str, tok):
    print(f"\n=== {tok_name}  ({tok.model_max_length} ctx) ===")

    prompt_text = tok.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_ids = tok.encode(prompt_text)
    print(f"prompt len = {len(prompt_ids)}")

    prev_ids = prompt_ids
    for i, msg in enumerate(completion_msgs):
        prefix_conv = prompt_msgs + completion_msgs[: i + 1]
        prefix_text = tok.apply_chat_template(
            prefix_conv,
            tokenize=False,
            add_generation_prompt=False,   # <- same as Environment code
            continue_final_message=True
        )
        curr_ids = tok.encode(prefix_text)

        # check prefix property (Environment does len(prev_ids)-1)
        ok = curr_ids[: len(prev_ids)] == prev_ids
        label = "✓" if ok else "✗ MISMATCH"

        print(f"step {i} | role={msg['role']:9} | total_len={len(curr_ids):4} "
              f"| new_tokens={len(curr_ids) - len(prev_ids):3} | {label}")

        # if not ok:
        #     # show the first mismatch for manual inspection
        #     for p, c in zip(prev_ids, curr_ids):
        #         if p != c:
        #             idx = curr_ids.index(c)
        #             print(" first diff at position", idx)
        #             print(" prev tail ids:", prev_ids[idx: idx + 10])
        #             print(" curr tail ids:", curr_ids[idx: idx + 10])
        #             break
        # ----- NEW: show exactly what was appended -----
        added_ids   = curr_ids[len(prev_ids):]
        added_text  = tok.decode(added_ids, skip_special_tokens=False)
        print("   added ids  :", added_ids)
        print("   added text :", repr(added_text))

        if not ok:
            # pinpoint the first differing position
            for j, (p, c) in enumerate(zip(prev_ids, curr_ids)):
                if p != c:
                    print("   first diff at token index", j)
                    print("   prev tail ids:", prev_ids[j: j + 10])
                    print("   curr tail ids:", curr_ids[j: j + 10])
                    break

        prev_ids = curr_ids


def main():
    for tag, name in MODELS.items():
        tok = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True
        )
        run_debug(name, tok)


if __name__ == "__main__":
    main()