#!/bin/sh
set -eu

if [ "$#" -ne 4 ]; then
    echo "Usage: sh bench/prefix_ab.sh EDGE_FALLBACK CHECKPOINT PREFIX_TOKENS PAIRS" >&2
    exit 2
fi
binary=$1
model=$2
prefix=$3
pairs=$4
case "$prefix" in 0*|''|*[!0-9]*) exit 2 ;; esac
case "$pairs" in 0*|''|*[!0-9]*) exit 2 ;; esac
export LC_ALL=C
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
trap 'exit 130' HUP INT TERM

jq -nc '
    def q($a; $b): "Reply with just the number. What is \($a) plus \($b)?";
    def handbook($office; $question):
        "\($office) office service handbook. Use it for internal requests.\n"
        + ([range(1; 111) | "Service \(.) uses the weekday schedule and records all changes.\n"] | join(""))
        + $question;
    [handbook("North"; q(2; 2)), handbook("North"; q(3; 3)),
     handbook("North"; q(4; 4)), q(1; 1),
     handbook("South"; q(2; 2)), handbook("South"; q(3; 3)),
     handbook("North"; q(4; 4)), handbook("North"; q(2; 2))]
    | .[] | {prompt: .}' > "$scratch/requests.jsonl"

run() {
    "$binary" --batch --offline --model "$model" --max-tokens 8 --trace-token-ids "$@"         < "$scratch/requests.jsonl"
}

printf '%s\n'     "pair,first,phase,prompt_tokens,generated_tokens,reused_prefix_tokens,retained_prefix_bytes,baseline_prefill_ms,cached_prefill_ms,baseline_generation_ms,cached_generation_ms,baseline_load_ms,cached_load_ms,generated_ids"     > "$scratch/paired.csv"
pair=1
while [ "$pair" -le "$pairs" ]; do
    if [ "$((pair % 2))" -eq 1 ]; then
        first=baseline
        run > "$scratch/baseline.jsonl"
        run --prefix-cache-tokens "$prefix" > "$scratch/cached.jsonl"
    else
        first=cached
        run --prefix-cache-tokens "$prefix" > "$scratch/cached.jsonl"
        run > "$scratch/baseline.jsonl"
    fi
    jq -ner --slurpfile baseline "$scratch/baseline.jsonl" --slurpfile cached "$scratch/cached.jsonl"         --argjson pair "$pair" --arg first "$first" --argjson prefix "$prefix" '
        def time_ms: type == "number" and isfinite and . >= 0;
        def valid_metrics($answer; $row):
            ([.prefill_ms, .generation_ms, .load_ms] | all(time_ms))
            and .generation_ms >= .prefill_ms
            and .prompt_tokens > 0
            and .generated_tokens == ($answer.generated_token_ids | length)
            and .model_reused == ($row != 0)
            and (if $row == 0 then .load_ms > 0 else .load_ms == 0 end);
        ["warmup", "hit", "hit", "short", "replace", "hit", "replace", "hit"] as $phases
        | if ($baseline | length) != ($phases | length) or ($cached | length) != ($phases | length)
          then error("wrong number of generated rows") else . end
        | range(0; $phases | length) as $i
        | $baseline[$i] as $a | $cached[$i] as $b
        | $a.local_metrics as $x | $b.local_metrics as $y
        | (if $phases[$i] == "hit" then $prefix else 0 end) as $reuse
        | if $a.ok != true or $b.ok != true or $a.source != "mlx" or $b.source != "mlx"
             or $a.line != ($i + 1) or $b.line != ($i + 1)
             or $a.generated_token_ids != $b.generated_token_ids
             or ($a.generated_token_ids | length) == 0 or $a.text != $b.text
             or (($x | valid_metrics($a; $i)) | not)
             or (($y | valid_metrics($b; $i)) | not)
             or $x.prompt_tokens != $y.prompt_tokens
             or $x.reused_prefix_tokens != 0 or $x.retained_prefix_bytes != 0
             or $y.reused_prefix_tokens != $reuse or $y.retained_prefix_tokens != $prefix
             or $y.retained_prefix_bytes <= 0
          then error("token or cache accounting mismatch in pair \($pair), case \($i + 1)")
          else [$pair, $first, $phases[$i], $y.prompt_tokens, $y.generated_tokens,
                $y.reused_prefix_tokens, $y.retained_prefix_bytes, $x.prefill_ms, $y.prefill_ms,
                $x.generation_ms, $y.generation_ms, $x.load_ms, $y.load_ms,
                ($a.generated_token_ids | map(tostring) | join(" "))] | @csv end'         >> "$scratch/paired.csv"
    pair=$((pair + 1))
done
cat "$scratch/paired.csv"
