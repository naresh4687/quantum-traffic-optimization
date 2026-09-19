# Featherless AI explanation layer (Phase 7)

Featherless AI is used **only to explain results that the deterministic system has already computed**. It does
not choose signal plans, change QUBO coefficients or QAOA parameters, optimize traffic, override the emergency
controller, or act as a source of simulation truth. If Featherless is unavailable the application behaves
exactly the same and prints a deterministic explanation built from the same numbers.

## Architecture

```
Traffic simulator
      |
Controllers / QUBO / QAOA          (deterministic; the source of truth)
      |
Measured results  (saved CSV / JSON)
      |
Structured analysis context        (small, whitelisted, JSON)
      |                    \
Featherless (optional)      Deterministic fallback (no model, always available)
      |                    /
Natural-language explanation
```

The dependency runs one way. `qtraffic.ai` imports nothing from the simulator, controllers, optimization or
emergency code, and none of that code imports `qtraffic.ai` (both are tested). Only
`src/qtraffic/ai/featherless.py` touches the OpenAI-compatible SDK and the endpoint.

`explain_signal_choice(context, "I3")` does not decide what I3 should do. It reports the plan that the
optimization pipeline already computed for I3 (present in the context), or says that no computed plan is
available.

## Configuration (environment variables only)

| Variable | Required | Meaning |
|---|---|---|
| `FEATHERLESS_API_KEY` | for any API call | API key. Never hard-coded, never logged, excluded from `repr`, redacted from errors. |
| `FEATHERLESS_MODEL` | for any API call | The model to use. **No default is assumed**: set the model supplied by the hackathon / your Featherless account. |
| `FEATHERLESS_BASE_URL` | no | Default `https://api.featherless.ai/v1`. |

`.env.example` is a template with an empty key; copy it to `.env` (which is git-ignored) or export the
variables in your shell. The project does not auto-load `.env`: load it with your shell or IDE. Never commit a
filled-in `.env`.

Fixed defaults in code: request timeout 20 s, `max_tokens` 400, temperature 0.2, no SDK retries, one request per
explanation, and successful results are cached in memory so the same question is not asked twice.

Dependency: `openai>=1.40,<3` (in `requirements.txt` and the optional `ai` extra). It is imported lazily; if it
is missing, the layer reports `package_missing` and uses the fallback.

## Structured analysis context

`TrafficAnalysisContext` holds only data derived from the simulator and the saved results: scenario, controller,
baseline, seeds, horizon; traffic (admitted / rejected / exited, throughput, waiting and waiting per admitted
vehicle, average / max / final queue, blocked); signal (plan changes, NS green share, computed plans);
baseline comparison (percentage changes on identical demand); environmental **proxy** (fuel, CO2, assumed
coefficients, labelled as a simulation proxy); optimization (QUBO energy); QAOA (depth, gates, sampled and exact
energies, feasible rate, optimum probability, statevector values, backend); emergency (travel time, delay,
stops, normal-traffic impact, restoration check). Standard limitations are included.

The context is whitelisted: unknown keys, non-finite numbers, oversized strings and path-like strings are
rejected, so raw logs, file paths, credentials and environment variables cannot reach a prompt. The numbers in
the context stay authoritative: `ExplanationResult.context` always carries them, whatever text a model returns.
Contexts are built from real saved results by `load_controller_context`, `load_qaoa_context` and
`load_emergency_context`.

## What is sent

A fixed system prompt plus one user message containing a one-line task and the context as compact JSON. Measured
sizes for the demo contexts: the context is 1.2 to 1.8 KB and the whole prompt about 2.5 to 3.1 KB. The system prompt tells the model to use only the supplied results, not to invent numbers or
experiments, not to choose or change signal plans or any control parameter, not to claim quantum advantage, not
to claim real-world emissions, to separate measured outputs from proxies, and to keep to 100 to 250 words.
Nothing else is sent: no API key, environment variables, file paths, CSV files or logs.

## Deterministic fallback

Used when there is no key, no model, an invalid base URL, a missing SDK, any request error, or a response that
fails validation. It is a pure function of the context (same input, same text, no network, no model), it uses the
actual values, and it always states the limitations. The returned object shows what happened:

```
{"available": false, "source": "deterministic_fallback", "explanation": "...", "fallback_reason": "missing_api_key", ...}
```

`fallback_reason` is one of: `missing_api_key`, `missing_model`, `invalid_base_url`, `package_missing`,
`authentication`, `rate_limit`, `timeout`, `connection`, `server_error`, `malformed_response`,
`validation_failed`, `request_failed`. With no key or model, no client is built and nothing is sent.

## Error handling

Authentication failure (401/403), rate limit (429), timeout, connection failure, server error (5xx), a malformed
response (no `choices[0].message.content`, or non-text content) and any other exception are all converted to a
fallback explanation. A failure never propagates to the simulator, optimizers or emergency controller.

## Response validation (lightweight, not a fact-check)

Text must exist, be a string, be 60 to 3000 characters, contain no API key or credential-like text, no
environment-variable names and no filesystem paths, and contain the expected content for its kind (a QAOA
explanation must mention the simulator, an environment explanation the proxy, an emergency explanation the
emergency). Sentences that assert a quantum advantage or speedup, or a real-world emissions change, or an
emissions change without the proxy/estimate qualifier, are rejected unless negated. Invalid text is discarded
in favour of the deterministic explanation. The numbers are never checked against the prose: they live in the
context.

## Privacy and data minimization

Only the whitelisted, summarized experiment context is sent. The key is redacted from every error message and
never appears in results, logs or prompts. Prompts are small and are not re-sent for the same result.

## Wording requirements

QAOA: "QAOA was evaluated using a classical quantum-circuit simulator (Qiskit Aer). The experiment measures
solution quality and sampling behavior, not quantum hardware speedup." Exact enumeration is the ground truth.

Environment: "Under the configured waiting-based simulation proxy, estimated CO2 changed by X%." Fuel and CO2
are linear in waiting time, are not measurements, and do not model acceleration, speed, vehicle type, engine
efficiency or NOx/PM emissions. See `docs/metrics_and_environmental_model.md`.

## Running

Without Featherless (works out of the box):

```
python scripts/demo_featherless.py
```

This loads real saved results (Phases 3 to 6), builds contexts, tries Featherless if configured, falls back
otherwise, and prints each explanation with its `source` (`featherless` or `deterministic_fallback`).

With Featherless (PowerShell shown; use `export` in bash):

```
$env:FEATHERLESS_API_KEY = "<your key>"
$env:FEATHERLESS_MODEL = "<the model supplied for the project>"
python scripts/demo_featherless.py
```

## Local smoke test (optional, live, not part of pytest)

```
python scripts/test_featherless.py
```

Requires `FEATHERLESS_API_KEY` and `FEATHERLESS_MODEL`. It sends one small request with a tiny synthetic context
and prints success or failure. Without a key it prints that the smoke test was skipped. It never prints the key.

## Tests

`tests/test_featherless_ai.py` uses a fake client and a fake SDK only, and blocks network connections for the
whole module, so a test cannot reach Featherless. It covers configuration, the missing-key / model / URL paths,
serialization, deterministic fallbacks for each explanation kind, mocked success, timeout, authentication,
rate-limit and malformed responses, secret leakage, the exact content of the prompt, and that numbers stay
authoritative.
