### Removed

* Removed the library helper `lmterminal.gpt_integration.estimate_prompt_cost`.
  Use `lmterminal.estimation.estimate_request(prepare_request(model, messages)).input_cost_usd`
  for numeric `Decimal` USD (or `None` when unavailable), importing `prepare_request`
  from `lmterminal.request_options`.
