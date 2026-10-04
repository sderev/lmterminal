### Changed

* Label `--tokens` as a local heuristic for Standard uncached input, preserve small
  USD amounts, and explain cache discounts and excluded charges.
* Report partial or unavailable estimates for tools, schemas, unsupported messages,
  unknown tokenizers/prices and nonstandard service tiers. First use may download
  public tokenizer data; loading failures now explain how to retry.

### Fixed

* Treat special-token-looking literals as ordinary text during estimation.
* Validate effective model and controls before both estimation and generation,
  including aliases and invalid model names selected from templates.

### Removed

* Replace `gpt_integration.num_tokens_from_string`, `num_tokens_from_messages`,
  `estimated_cost`, `PromptCostEstimate` and `estimate_prompt_cost_details` with
  `estimation.estimate_request` and its `InputEstimate` result. Use
  `request_options.prepare_request` to prepare messages/model/controls; costs are
  numeric `Decimal` USD or `None`, with explicit reasons for unavailable estimates.
