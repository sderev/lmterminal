## Changed

* Require `tiktoken` 0.14.0 and resolve each canonical model directly through
  upstream tokenizer lookup. GPT-6 estimates remain unavailable until upstream
  recognizes those IDs; tokenizer data failures retain cache/download guidance.
