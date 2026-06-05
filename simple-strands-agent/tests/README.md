# SSA test suite

25 tests (36 with parametrization) covering the load-bearing paths of the SSA
harness: environment execution, tools, conversation management, model-response
policy hooks, agent retry logic, git ops, dataset config, and two end-to-end
agent runs driven by a deterministic fake model.

Run the suite with:

```bash
uv run pytest tests/
```

## Shared infrastructure

- [tests/conftest.py](conftest.py) — `make_cfg()` helper, `FakeEnvironment`
  (in-memory `Environment`), `ShellExecutingEnvironment` (runs real bash in a
  fixed workdir), and the `git_repo` fixture.
- [tests/fake_model.py](fake_model.py) — `FakeSSAModel` + `FakeTurn`. Replays
  a fixed sequence of Bedrock-shaped streaming events so tests can drive a
  real `StrandsResolverAgent` without hitting any API.

---

## 1. Streaming exec & LocalEnvironment

File: [tests/environments/test_streaming_exec.py](environments/test_streaming_exec.py)

| # | Test | Verifies |
|---|---|---|
| 1 | [`test_streaming_exec_basic`](environments/test_streaming_exec.py) | [`run_with_streaming_capture`](../src/ssa/environments/_streaming_exec.py) runs a fast command, returns exit code 0 and captures stdout. |
| 2 | [`test_streaming_exec_timeout_returns_124_with_partial`](environments/test_streaming_exec.py) | On `timeout_sec` expiry, [`run_with_streaming_capture`](../src/ssa/environments/_streaming_exec.py) returns exit code 124 **and** preserves output produced before the timeout. |
| 3 | [`test_headtail_buffer_drops_middle`](environments/test_streaming_exec.py) | [`HeadTailBuffer`](../src/ssa/environments/_streaming_exec.py) keeps head + tail and evicts middle bytes, inserting a `< ... N bytes dropped ... >` marker on `materialize()`. |

File: [tests/environments/test_local.py](environments/test_local.py)

| # | Test | Verifies |
|---|---|---|
| 4 | [`test_local_env_execute_basic`](environments/test_local.py) | [`LocalEnvironment.execute_bash`](../src/ssa/environments/environment.py) returns the expected dict shape (`status`, `exit_code`, `output`, `error`, `command`) for a successful command. |
| 5 | [`test_local_env_execute_failure`](environments/test_local.py) | A non-zero exit code produces `status="error"` and `error == output` (the code surfaces the full output as the error payload). |
| 6 | [`test_local_env_timeout_captures_partial`](environments/test_local.py) | A command that prints before sleeping past the deadline returns exit code 124 with partial stdout preserved, and — importantly — `error == ""` (SSA treats 124 as not-an-error). |

## 2. Tool layer

File: [tests/tools/test_bash.py](tools/test_bash.py)

| # | Test | Verifies |
|---|---|---|
| 7 | [`test_bash_missing_command`](tools/test_bash.py) | [`bash`](../src/ssa/tools/bash.py) returns `status="error"` when the tool input has no `command`, and never dispatches to the environment. |
| 8 | [`test_bash_timeout_124_message_and_partial`](tools/test_bash.py) (2 params) | Exit code 124 prepends the `"Command timed-out with limit"` notice; partial output is preserved by default and **cleared** when `publish_partial_output=False`. |
| 9 | [`test_bash_output_clipping`](tools/test_bash.py) | Output exceeding `MAX_LINES_LIMIT` (250) is head/tail clipped with a `"lines clipped"` marker; head line `line-0` and tail `line-499` survive, middle does not. |

File: [tests/tools/test_batch_bash.py](tools/test_batch_bash.py)

| # | Test | Verifies |
|---|---|---|
| 10 | [`test_batch_bash_ignore_errors_false_stops`](tools/test_batch_bash.py) | With `ignore_errors=False`, [`batch_bash`](../src/ssa/tools/batch_bash.py) halts after the first failure, emits `"N remaining command(s) skipped"`, and doesn't dispatch the remaining commands. |

File: [tests/tools/test_submit.py](tools/test_submit.py)

| # | Test | Verifies |
|---|---|---|
| 11 | [`test_submit_validation`](tools/test_submit.py) (4 params) | [`submit`](../src/ssa/tools/submit.py) rejects missing `summary` / `status`, non-list `paths`, and paths that don't exist in the environment, each with a diagnostic error message. |
| 12 | [`test_submit_success_sets_state`](tools/test_submit.py) | A valid `submit` writes `stop_event_loop=True` and `submit_paths=[...]` into `request_state`, signalling the event loop to terminate. |

## 3. Conversation manager

File: [tests/conversation_manager/test_conversation_manager.py](conversation_manager/test_conversation_manager.py)

| # | Test | Verifies |
|---|---|---|
| 13 | [`test_apply_management_noop_under_window`](conversation_manager/test_conversation_manager.py) | [`AdaptiveConversationManager.apply_management`](../src/ssa/conversation_manager/conversation_manager.py) doesn't touch `agent.messages` when the list is shorter than `window_size`. |
| 14 | [`test_reduce_from_overflow_truncates_largest_tool_result`](conversation_manager/test_conversation_manager.py) | `reduce_context(from_overflow=True)` truncates the message with the **largest** tool-result payload (not the first one over the threshold), leaving small results untouched. |
| 15 | [`test_reduce_context_skips_orphan_toolresult_at_trim_index`](conversation_manager/test_conversation_manager.py) | When a naive `trim_index` would land on an orphan `toolResult` (no preceding `toolUse`), the trim advances past it so the retained window never starts with a dangling result. |

## 4. Hooks

File: [tests/hooks/test_content_hook.py](hooks/test_content_hook.py)

| # | Test | Verifies |
|---|---|---|
| 16 | [`test_content_hook_empty_content_throttles`](hooks/test_content_hook.py) (3 params) | [`ContentHook`](../src/ssa/hooks/content_hook.py) raises `ModelThrottledException` on empty content list regardless of `stop_reason` (`end_turn`, `tool_use`, `max_tokens`). |
| 17 | [`test_content_hook_max_tokens_with_tooluse_recovers`](hooks/test_content_hook.py) | On `stop_reason="max_tokens"` with a `toolUse` in content, the hook appends a recovered assistant message (no tool uses) + a user feedback message, then raises throttling so the agent will retry. |
| 18 | [`test_content_hook_gemini_patterns_throttle`](hooks/test_content_hook.py) (3 params) | Gemini-specific garbage in `end_turn` text (pure `<ctrl46>`, > 2 occurrences, `call:default_api:...<ctrl46>`) all trip throttling. |
| 19 | [`test_event_loop_limiter_enforces_limits`](hooks/test_content_hook.py) | [`EventLoopLimiterHook`](../src/ssa/hooks/content_hook.py) raises `MaxRecursionsReachedException` once the recursion counter exceeds `max_recursion_length`, and sets `event.terminate=True` once the loop counter exceeds `max_loop_length`. |

## 5. Agent retry loop

File: [tests/agent/test_agent_retry.py](agent/test_agent_retry.py)

| # | Test | Verifies |
|---|---|---|
| 20 | [`test_agent_context_overflow_retries_with_reduce`](agent/test_agent_retry.py) | [`StrandsResolverAgent._execute_event_loop_cycle`](../src/ssa/agent.py) catches `ContextWindowOverflowException`, calls `conversation_manager.reduce_context(from_overflow=True)`, and retries. The event from the retry reaches the caller and `AgentCompletedEvent` fires in `finally`. |
| 21 | [`test_agent_max_tokens_retries_with_last_message_popped`](agent/test_agent_retry.py) | On `MaxTokensReachedException`, the agent pops `messages[-1]` and retries the event loop. |

## 6. Git ops & dataset config

File: [tests/utils/test_git_ops.py](utils/test_git_ops.py)

| # | Test | Verifies |
|---|---|---|
| 22 | [`test_get_git_patch_captures_modified_and_new_files`](utils/test_git_ops.py) | [`get_git_patch`](../src/ssa/utils/git_ops.py) returns a diff containing both modified and newly-added files; the optional `paths=[...]` filter restricts output to only the listed paths. |

File: [tests/utils/test_handle_config.py](utils/test_handle_config.py)

| # | Test | Verifies |
|---|---|---|
| 23 (a) | [`test_tb2_raises_without_env_var`](utils/test_handle_config.py) | [`get_problem_statement_for_tb2`](../src/ssa/utils/handle_config.py) raises `ValueError` when `TB2_INSTRUCTIONS_MAP` is unset. |
| 23 (b) | [`test_tb2_missing_identifier_raises`](utils/test_handle_config.py) | TB2 path raises `KeyError` when the identifier isn't in the instructions map. |
| 23 (c) | [`test_unknown_dataset_raises`](utils/test_handle_config.py) | [`identifier_to_problem_statement`](../src/ssa/utils/handle_config.py) raises `ValueError` on unsupported dataset names. |
| 23 (d) | [`test_sbv_uses_local_cache_when_available`](utils/test_handle_config.py) | [`get_problem_statement_for_swebench`](../src/ssa/utils/handle_config.py) uses the local cached HF dataset (via `load_from_disk`) when `HF_SBV_DATASET_OFFLINE_LOCATION` points at an existing path. |

## 7. End-to-end

File: [tests/e2e/test_end_to_end.py](e2e/test_end_to_end.py)

| # | Test | Verifies |
|---|---|---|
| 24 | [`test_e2e_local_agent_runs_to_submit`](e2e/test_end_to_end.py) | Full wiring: [`StrandsResolverAgent`](../src/ssa/agent.py) + [`LocalEnvironment`](../src/ssa/environments/environment.py) + [`TrajectoryHook`](../src/ssa/hooks/traj_hook.py) + real [`bash`](../src/ssa/tools/bash.py) & [`submit`](../src/ssa/tools/submit.py) tools, driven by [`FakeSSAModel`](fake_model.py). Agent runs `echo > file`, submits, and `trajectory.json` reflects the conversation. The bash side-effect is verified on disk and `result.state.submit_paths` is populated. |
| 25 | [`test_e2e_throttling_retry_path`](e2e/test_end_to_end.py) | [`ContentHook`](../src/ssa/hooks/content_hook.py) converts an empty-content assistant turn into a throttle; the agent retries and the second turn's `submit` succeeds. `model.call_count >= 2` proves the retry path ran end-to-end. |
