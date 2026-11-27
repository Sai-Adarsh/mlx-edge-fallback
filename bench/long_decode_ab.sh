#!/bin/sh
set -eu

if [ "$#" -ne 4 ]; then
    echo "Usage: sh bench/long_decode_ab.sh EDGE_FALLBACK CHECKPOINT PREFIX_TOKENS PAIRS" >&2
    exit 2
fi
binary=$1
model=$2
prefix=$3
pairs=$4
case "$prefix" in 0*|''|*[!0-9]*) exit 2 ;; esac
case "$pairs" in 0*|''|*[!0-9]*) exit 2 ;; esac
jq_bin="${EDGE_JQ:-jq}"
export LC_ALL=C
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
trap 'exit 130' HUP INT TERM

run() {
    "$binary" --batch --offline --model "$model" --trace-token-ids "$@" \
        < "$scratch/requests.jsonl"
}

printf '%s\n' \
    "pair,first,position,case,prompt_tokens,generated_tokens,cached_prefix_tokens,retained_prefix_bytes,baseline_load_ms,cached_load_ms,baseline_ttft_ms,cached_ttft_ms,baseline_decode_ms,cached_decode_ms,baseline_decode_tok_s,cached_decode_tok_s,baseline_generation_ms,cached_generation_ms,generated_ids" \
    > "$scratch/paired.csv"
pair=1
while [ "$pair" -le "$pairs" ]; do
    # Cross the two process orders with the two prompt-length orders.
    reverse=$(( ((pair - 1) / 2) % 2 ))
    "$jq_bin" -nc --argjson reverse "$reverse" '
        def request($rows; $budget):
            {prompt:
                ("North office service handbook. This fictional background is for the writing exercise below.\n"
                + ([range(1; $rows + 1) | "Service \(.) uses the weekday schedule and records all changes.\n"] | join(""))
                + "Write a 200-word story about a lost robot finding its way back to a small lighthouse. Use plain English and describe three concrete events. Start directly with the story."),
             max_tokens: $budget};
        request(110; 8),
        (if $reverse == 0 then request(220; 48), request(380; 48)
         else request(380; 48), request(220; 48) end)' > "$scratch/requests.jsonl"

    if [ "$((pair % 2))" -eq 1 ]; then
        first=baseline
        run > "$scratch/baseline.jsonl"
        run --prefix-cache-tokens "$prefix" > "$scratch/cached.jsonl"
    else
        first=cached
        run --prefix-cache-tokens "$prefix" > "$scratch/cached.jsonl"
        run > "$scratch/baseline.jsonl"
    fi

    "$jq_bin" -ner --slurpfile baseline "$scratch/baseline.jsonl" --slurpfile cached "$scratch/cached.jsonl" \
        --argjson pair "$pair" --arg first "$first" --argjson prefix "$prefix" --argjson reverse "$reverse" '
        def positive: type == "number" and isfinite and . > 0;
        def valid($reply; $pos; $budget):
            $reply.ok == true and $reply.source == "mlx" and $reply.line == ($pos + 1)
            and ($reply.generated_token_ids | length) == $budget
            and $reply.local_metrics.generated_tokens == $budget
            and $reply.local_metrics.stop_reason == "token_limit"
            and $reply.local_metrics.stop_decision_ms == null
            and ($reply.local_metrics | [.ttft_ms, .decode_ms, .decode_tokens_per_second, .generation_ms] | all(positive))
            and ($reply.local_metrics | .prefill_ms == .ttft_ms
                and (.decode_ms * .decode_tokens_per_second - (($budget - 1) * 1000) | fabs < 0.0001)
                and .ttft_ms + .decode_ms <= .generation_ms + 0.000001
                and .model_reused == ($pos > 0)
                and (if $pos == 0 then (.load_ms | positive) else .load_ms == 0 end));
        if ($baseline | length) != 3 or ($cached | length) != 3
        then error("expected a seed and two long requests") else . end
        | range(0; 3) as $i
        | $baseline[$i] as $a | $cached[$i] as $b
        | $a.local_metrics as $x | $b.local_metrics as $y
        | (if $i == 0 then "seed" elif ($i == 1 and $reverse == 0) or ($i == 2 and $reverse == 1) then "220-lines" else "380-lines" end) as $case
        | (if $i == 0 then 8 else 48 end) as $budget
        | (if $i == 0 then 0 else $prefix end) as $reuse
        | if (valid($a; $i; $budget) | not) or (valid($b; $i; $budget) | not)
            or $a.generated_token_ids != $b.generated_token_ids or $a.text != $b.text
            or $x.prompt_tokens != $y.prompt_tokens
            or $y.prompt_tokens <= $prefix or ($i > 0 and $y.prompt_tokens < 2000)
            or $x.reused_prefix_tokens != 0 or $x.retained_prefix_bytes != 0
            or $y.reused_prefix_tokens != $reuse or $y.retained_prefix_tokens != $prefix
            or $y.retained_prefix_bytes <= 0
          then error("token, timing, or cache mismatch in pair \($pair), case \($case)")
          else [$pair, $first, $i + 1, $case, $x.prompt_tokens, $budget, $reuse, $y.retained_prefix_bytes,
                $x.load_ms, $y.load_ms, $x.ttft_ms, $y.ttft_ms, $x.decode_ms, $y.decode_ms,
                $x.decode_tokens_per_second, $y.decode_tokens_per_second, $x.generation_ms, $y.generation_ms,
                ($a.generated_token_ids | map(tostring) | join(" "))] | @csv end' >> "$scratch/paired.csv"
    printf "Validated pair %s/%s\n" "$pair" "$pairs" >&2
    pair=$((pair + 1))
done
cat "$scratch/paired.csv"
