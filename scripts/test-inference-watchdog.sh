#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/inference-watchdog.sh"

assert_json() { jq -e "$2" <<< "$1" >/dev/null; }
sample() { printf '{"generation_tokens_total":%s,"prompt_tokens_total":1000,"request_success_total":3,"num_requests_running":%s}' "${1:-100}" "${2:-2}"; }
step() { observe "$1" invocation 2000 "$2" "$(sample "${3:-100}" "${4:-2}")" "${5:-true}" "${6:-false}"; }
state_of() { jq -c .state <<< "$1"; }

test_metrics() {
  local text valid bad
  text=$'vllm:num_requests_running{engine="0"} 2\nvllm:generation_tokens_total{engine="0"} 100\nvllm:prompt_tokens_total{engine="0"} 1000\n'
  valid="$(parse_metrics <<< "$text")"
  assert_json "$valid" '.num_requests_running == 2 and .prompt_tokens_total == 1000'
  valid="$(parse_metrics <<< "${text}vllm:generation_tokens_total{engine=\"1\"} 50")"
  assert_json "$valid" '.generation_tokens_total == 150'
  if parse_metrics <<< '' >/dev/null 2>&1; then return 1; fi
  for bad in NaN Inf -1 10bad 1e999; do
    if parse_metrics <<< "${text}vllm:prompt_tokens_total $bad" >/dev/null 2>&1; then return 1; fi
  done
}
test_gpu_fault() {
  local first decision
  first="$(step '{}' 1000 100 2 true true)"
  assert_json "$first" '.reason == null'
  decision="$(step "$(state_of "$first")" 1059)"
  assert_json "$decision" '.reason == null'
  decision="$(step "$(state_of "$first")" 1060)"
  assert_json "$decision" '(.reason | contains("GPU")) and .state.restarts == [1060]'
}
test_prefill_and_progress() {
  local first decision
  first="$(step '{}' 1000)"
  decision="$(step "$(state_of "$first")" 1599)"
  assert_json "$decision" '.reason == null'
  decision="$(step "$(state_of "$first")" 1600)"
  assert_json "$decision" '.reason | contains("10 minutes")'
  first="$(step '{}' 1000 100 2 true true)"
  decision="$(step "$(state_of "$first")" 1060 101)"
  assert_json "$decision" '.reason == null and .state.gpu_fault == false'
}
test_idle() {
  local first decision
  first="$(step '{}' 1000 100 0 true true)"
  decision="$(step "$(state_of "$first")" 3000 100 0)"
  assert_json "$decision" '.reason == null'
}
test_startup() {
  local first decision
  first="$(observe '{}' new 200 1000 null false true)"
  assert_json "$first" '.reason == null'
  decision="$(observe "$(state_of "$first")" new 899 1699 null false false)"
  assert_json "$decision" '.reason == null'
  decision="$(observe "$(state_of "$first")" new 900 1700 null false false)"
  assert_json "$decision" '.reason | contains("unavailable")'
}
test_api_outage() {
  local first decision
  first="$(observe '{}' new 900 1000 null false false)"
  decision="$(observe "$(state_of "$first")" new 1499 1599 null false false)"
  assert_json "$decision" '.reason == null'
  decision="$(observe "$(state_of "$first")" new 1500 1600 null false false)"
  assert_json "$decision" '.reason | contains("unavailable")'
}
test_new_invocation() {
  local first decision
  first="$(step '{}' 1000 100 2 true true | jq -c '.state | .restarts = [900]')"
  decision="$(observe "$first" new 2000 2000 "$(sample)" true false)"
  assert_json "$decision" '.reason == null and .state.progress_at == 2000 and .state.restarts == [900]'
}
test_restart_limits() {
  local first state decision
  first="$(step '{}' 1000 100 2 true true)"
  state="$(jq -c '.state | .restarts = [950]' <<< "$first")"
  decision="$(step "$state" 1100)"
  assert_json "$decision" '.reason == null'
  decision="$(step "$state" 1850)"
  assert_json "$decision" '.reason != null'
  state="$(jq -c '.state | .restarts = [100,200,300]' <<< "$first")"
  decision="$(step "$state" 1600)"
  assert_json "$decision" '.reason == null'
  decision="$(step "$state" 4000)"
  assert_json "$decision" '.reason != null'
}
test_any_hour() {
  local now first decision
  for now in 1794481200 1794519900 1794522600; do
    first="$(step '{}' "$((now-600))")"
    decision="$(step "$(state_of "$first")" "$now")"
    assert_json "$decision" '.reason != null'
  done
}
test_state_compatibility() {
  local path="$WATCHDOG_TEST_ROOT/state/inference-watchdog.json" legacy bad decision
  legacy='{"invocation":"invocation","seen_at":1000.25,"progress_at":1000,"restarts":[900],"busy":true,"gpu_fault":true,"signature":[100,1000,3]}'
  save_state "$path" "$legacy"
  [[ "$(stat -c %a "$path")" == 600 ]]
  [[ "$(stat -c %a "$(dirname "$path")")" == 700 ]]
  assert_json "$(cat "$path")" '.seen_at == 1000.25 and .restarts == [900]'
  decision="$(step "$(cat "$path")" 1060)"
  assert_json "$decision" '.reason == null and .state.restarts == [900]'
  for bad in '[]' '{"restarts":false}' '{"busy":2}' '{"seen_at":null}'; do
    decision="$(step "$bad" 1000)"
    assert_json "$decision" '.reason == null and .state.restarts == []'
  done
}
test_command_timeout() {
  local started=$SECONDS output
  output="$(run 1 printf '%s' 'literal $(echo no); quote')"
  [[ "$output" == 'literal $(echo no); quote' ]]
  if run 1 bash -c 'exit 7'; then return 1; fi
  if run 0.05 bash -c 'sleep 5'; then return 1; fi
  ((SECONDS - started < 2))
}

# Exercise the real main function, state persistence and diagnostics with fake
# systemctl/journalctl/HTTP only. No real model or service requests are issued.
fixture() {
  export HOME="$WATCHDOG_TEST_ROOT/$1"
  mkdir -p "$HOME"
  CALLS="$HOME/calls.log"
  : > "$CALLS"
  FIXTURE_ACTIVE=active FIXTURE_CURRENT=invocation FIXTURE_FAULT=false FIXTURE_RUNNING=2
  run() {
    local limit="$1" program="$2"; shift 2
    printf '%s %s\n' "$program" "$*" >> "$CALLS"
    if [[ "$program" == journalctl ]]; then
      if [[ "$FIXTURE_FAULT" == true ]]; then printf 'Timedout job: VLLM::EngineCor\n'; fi
    elif [[ "$program" == systemctl && "$*" == *ActiveState* ]]; then
      printf 'ActiveState=%s\nInvocationID=invocation\nExecMainStartTimestamp=@%s\n' "$FIXTURE_ACTIVE" "$(($(date +%s)-2000))"
    elif [[ "$program" == systemctl && "$*" == *try-restart* ]]; then
      [[ "$limit" == 90 ]]
    elif [[ "$program" == systemctl ]]; then printf '%s\n' "$FIXTURE_CURRENT"
    else return 99; fi
  }
  fetch() {
    printf 'HTTP %s\n' "$1" >> "$CALLS"
    if [[ "$1" == /metrics ]]; then
      printf 'vllm:num_requests_running %s\nvllm:generation_tokens_total 100\nvllm:prompt_tokens_total 1000\nvllm:request_success_total 3\n' "$FIXTURE_RUNNING"
    else [[ "$1" == /health ]]; fi
  }
}
seed_stall() {
  local now
  now="$(date +%s)"
  save_state "$HOME/.local/state/b70-qwen38-vllm/inference-watchdog.json" \
    "$(step '{}' "$((now-${1:-601}))" | jq -c .state)"
}
test_main_recovery() {
  fixture recovery
  seed_stall
  main
  grep -q 'systemctl --user try-restart b70-qwen38-vllm.service' "$CALLS"
  assert_json "$(cat "$HOME/.local/state/b70-qwen38-vllm/inference-watchdog.json")" '(.restarts | length) == 1 and (.last_reason | contains("10 minutes"))'
  [[ "$(find "$HOME" -name state.json | wc -l)" == 1 ]]
}
test_main_race() {
  fixture race
  FIXTURE_CURRENT=new-invocation
  seed_stall
  main
  if grep -q try-restart "$CALLS"; then return 1; fi
}
test_main_stopped() {
  fixture stopped
  FIXTURE_ACTIVE=inactive
  main
  if grep -q 'HTTP|try-restart' "$CALLS"; then return 1; fi
  [[ ! -d "$HOME/.local" ]]
}
test_main_gpu() {
  fixture gpu
  FIXTURE_FAULT=true
  seed_stall 61
  main
  grep -q try-restart "$CALLS"
  assert_json "$(cat "$HOME/.local/state/b70-qwen38-vllm/inference-watchdog.json")" '.last_reason | contains("GPU")'
}
test_main_idle() {
  fixture idle
  FIXTURE_RUNNING=0
  seed_stall
  main
  if grep -q try-restart "$CALLS"; then return 1; fi
}

if [[ "${1:-}" == --case ]]; then "$2"; exit; fi
export WATCHDOG_TEST_ROOT
WATCHDOG_TEST_ROOT="$(mktemp -d /tmp/b70-watchdog-shell-test-XXXXXX)"
trap 'find "$WATCHDOG_TEST_ROOT" -depth -delete' EXIT
passed=0 failed=0
for case_name in test_metrics test_gpu_fault test_prefill_and_progress test_idle test_startup \
  test_api_outage test_new_invocation test_restart_limits test_any_hour test_state_compatibility \
  test_command_timeout test_main_recovery test_main_race test_main_stopped test_main_gpu test_main_idle; do
  if bash "$0" --case "$case_name"; then
    printf 'PASS %s\n' "$case_name"
    passed=$((passed+1))
  else
    printf 'FAIL %s\n' "$case_name" >&2
    failed=$((failed+1))
  fi
done
printf '%s passed, %s failed\n' "$passed" "$failed"
[[ "$failed" == 0 ]]
