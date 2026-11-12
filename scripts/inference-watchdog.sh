#!/usr/bin/env bash
# Recover stalled vLLM inference; systemd owns process crashes and operator stops.
set -euo pipefail

SERVICE=b70-qwen38-vllm.service
BASE_URL=http://127.0.0.1:8081

# Pure recovery decision. JSON state remains compatible with older watchdogs.
observe() {
  local previous="$1" invocation="$2" age="$3" now="$4" metrics="$5" healthy="$6" fault="$7"
  jq -cn --argjson previous "$previous" --arg invocation "$invocation" \
    --argjson age "$age" --argjson now "$now" --argjson metrics "$metrics" \
    --argjson healthy "$healthy" --argjson fault "$fault" '
    def finite: type == "number" and (isnan | not) and (isinfinite | not);
    def valid:
      . as $s | type == "object" and
      all(["seen_at", "progress_at", "unhealthy_at"][];
        . as $k | ($s | has($k) | not) or ($s[$k] | finite)) and
      all(["busy", "gpu_fault"][];
        . as $k | ($s | has($k) | not) or ($s[$k] | type == "boolean")) and
      ((has("invocation") | not) or (.invocation | type == "string")) and
      ((has("restarts") | not) or (.restarts | type == "array" and all(.[]; finite)));
    ($previous | if (try valid catch false) then . else {} end) |
    [(.restarts // [])[] | select($now - . >= 0 and $now - . < 3600)] as $restarts |
    (if .invocation != $invocation or $now < (.seen_at // $now) then {} else . end) |
    .invocation = $invocation | .seen_at = $now | .restarts = $restarts |
    if $age < 900 and $metrics == null and ($healthy | not) then
      .progress_at = $now | .gpu_fault = false | .unhealthy_at //= ($now - $age) |
      {state: ., reason: null}
    else
      (if $fault then .gpu_fault = true else . end) |
      (if $metrics != null then
        [$metrics.generation_tokens_total, $metrics.prompt_tokens_total,
         ($metrics.request_success_total // 0)] as $signature |
        (($metrics.num_requests_running + ($metrics.num_requests_waiting // 0)) > 0) as $busy |
        (if $signature != .signature or ($busy | not) then
          .progress_at = $now | if $fault then . else .gpu_fault = false end
         else . end) |
        .signature = $signature | .busy = $busy
       else .progress_at //= $now end) |
      (if $healthy then .unhealthy_at = $now else .unhealthy_at //= $now end) |
      ($now - (.progress_at // $now)) as $stalled |
      (if .busy and .gpu_fault and $stalled >= 60 then
         "vLLM GPU job fault and no inference progress for 60 seconds"
       elif .busy and $stalled >= 600 then
         "Active inference has made no progress for 10 minutes"
       elif ($healthy | not) and $now - .unhealthy_at >= 600 then
         "Model API unavailable beyond startup grace or the 10-minute outage budget"
       else null end) as $reason |
      (if ($restarts | length) >= 3 or
          (($restarts | length) > 0 and $now - $restarts[-1] < 900)
       then null else $reason end) as $reason |
      (if $reason != null then .restarts += [$now] | .last_reason = $reason else . end) |
      {state: ., reason: $reason}
    end'
}

parse_metrics() {
  jq -Rn '
    [inputs | capture("^vllm:(?<name>\\w+)(?:\\{[^}]*\\})?\\s+(?<value>\\S+)")? |
      select(.name as $n | ["num_requests_running", "num_requests_waiting",
        "generation_tokens_total", "prompt_tokens_total", "request_success_total"] | index($n)) |
      .value |= (tonumber | if isnan or isinfinite or . < 0 then error("Invalid metric") else . end)] |
    reduce .[] as $v ({}; .[$v.name] = ((.[$v.name] // 0) + $v.value)) |
    if has("num_requests_running") and has("generation_tokens_total") and has("prompt_tokens_total") and
       all(.[]; (isnan | not) and (isinfinite | not)) then . else error("Invalid progress metrics") end'
}

run() { local seconds="$1"; shift; timeout --kill-after=1s "${seconds}s" "$@"; }
fetch() { curl --noproxy '*' --fail --silent --show-error --max-time 3 --connect-timeout 3 \
  --max-filesize 2097152 "$BASE_URL$1"; }

# Private, atomic replacement; flush the file before publishing a decision.
save_state() (
  local path="$1" json="$2" directory temporary
  directory="$(dirname -- "$path")"
  umask 077
  mkdir -p -- "$directory"
  chmod 700 -- "$directory"
  temporary="$(mktemp "$directory/.watchdog.XXXXXX")"
  trap 'if [[ -f "$temporary" ]]; then unlink -- "$temporary"; fi' EXIT
  printf '%s\n' "$json" > "$temporary"
  sync -d -- "$temporary"
  mv -f -- "$temporary" "$path"
)

capture_failure() {
  local path="$1" state="$2" directory old
  directory="$(dirname -- "$path")/recovery-$(jq -r '.seen_at | floor' <<< "$state")"
  save_state "$directory/state.json" "$state"
  run 5 journalctl -k -n 200 --no-pager > "$directory/kernel.log" 2>&1 || true
  run 5 journalctl --user -u "$SERVICE" -n 200 --no-pager > "$directory/engine.log" 2>&1 || true
  local -a directories=()
  mapfile -d '' -t directories < <(find "$(dirname -- "$path")" -mindepth 1 -maxdepth 1 \
    -type d -name 'recovery-*' -print0 | sort -z)
  local index
  for ((index=0; index+8<${#directories[@]}; index++)); do
    old="${directories[index]}"
    # Only prune directories containing ordinary diagnostic files.
    if [[ -z "$(find "$old" -mindepth 1 -maxdepth 1 ! -type f -print -quit)" ]]; then
      find "$old" -mindepth 1 -maxdepth 1 -type f -delete
      rmdir -- "$old"
    fi
  done
}

main() {
  local info active='' invocation='' started='' key value
  info="$(LC_ALL=C run 5 systemctl --user show "$SERVICE" \
    -p ActiveState -p InvocationID -p ExecMainStartTimestamp)"
  while IFS='=' read -r key value; do
    case "$key" in
      ActiveState) active="$value" ;;
      InvocationID) invocation="$value" ;;
      ExecMainStartTimestamp) started="$value" ;;
    esac
  done <<< "$info"
  [[ "$active" == active ]] || return 0
  [[ -n "$invocation" && -n "$started" ]] || return 1
  local path="${HOME:?HOME is required}/.local/state/b70-qwen38-vllm/inference-watchdog.json"
  local now started_epoch age previous metrics healthy=false fault=false kernel decision state reason current
  now="$(date +%s)"
  # Use the service wall-clock timestamp, matching persisted progress times.
  # /proc/uptime includes suspend time and cannot be compared to systemd's
  # CLOCK_MONOTONIC timestamps without accounting for earlier suspends.
  started_epoch="$(LC_ALL=C date --date "$started" +%s)"
  age="$((now - started_epoch))"
  previous="$(jq -ce 'select(type == "object")' "$path" 2>/dev/null)" || previous='{}'
  metrics="$(fetch /metrics 2>/dev/null | parse_metrics 2>/dev/null)" || metrics=null
  if fetch /health >/dev/null 2>&1; then healthy=true; fi
  local since
  since="$(jq -nr --argjson now "$now" --argjson age "$age" '[$now - 180, $now - $age] | max | floor')"
  if kernel="$(run 5 journalctl -k --since "@$since" -o cat --no-pager | head -c 2097152)"; then
    local line
    while IFS= read -r line; do
      if [[ "$line" == *'Timedout job:'* && "$line" == *'VLLM::EngineCor'* ]]; then
        fault=true
        break
      fi
    done <<< "$kernel"
  fi
  decision="$(observe "$previous" "$invocation" "$age" "$now" "$metrics" "$healthy" "$fault")"
  state="$(jq -c .state <<< "$decision")"
  save_state "$path" "$state"
  reason="$(jq -r '.reason // empty' <<< "$decision")"
  [[ -n "$reason" ]] || return 0
  capture_failure "$path" "$state"
  jq -cn --arg reason "$reason" --argjson metrics "$metrics" '{action: "restart", reason: $reason, metrics: $metrics}'
  current="$(run 5 systemctl --user show "$SERVICE" -p InvocationID --value)"
  if [[ "$current" == "$invocation" ]]; then
    run 90 systemctl --user try-restart "$SERVICE"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
