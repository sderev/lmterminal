
# 2025-05-25

## Fixed

`lmt templates edit` and `lmt templates add` are now working correctly again.

# 2025-05-25

## Added

Models below are now available:

* `gpt-4.1`
* `gpt-4.1-2025-04-14`
* `gpt-4.1-mini`
* `gpt-4.1-mini-2025-04-14`
* `gpt-4.1-nano`
* `gpt-4.1-nano-2025-04-14`
* `gpt-4.5-preview`
* `o3`
* `o3-2025-04-16`
* `o3-mini`
* `o3-mini-2025-01-31`
* `o4-mini`
* `o4-mini-2025-04-16`

# 2025-01-11

## Added

Now that the `o1` model is out of its preview, there are new endpoints:
* `o1`
* `o1-2024-12-17`

## Fixed

* The `--tokens` option now work correctly for the `o1` models. As of now, they don't accept a `system` message in the prompt, which was causing an error when estimating the cost of the tokens input.

# 2024-11-21

## Added

* Add OpenAI models endpoints:
    * `gpt-4o-2024-11-20`
    * `o1-preview`
    * ` o1-preview-2024-09-12`
    * `o1-mini`
    * `o1-mini-2024-09-12`

# 2024-09-30

## Changed

* Update the cost for the input tokens of `gpt-4o`.

# 2024-08-14

## Added

* Add `chatgpt-4o-latest` model (see: <https://platform.openai.com/docs/models/gpt-4o>).

* Add the `gpt-4o-2024-08-06` model.

# 2024-07-19

## Added

* Introduced `lmt models` command to output all valid model names and their aliases.

* Add `gpt-4o-mini`

## Changed

* Set `gpt-4o-mini` as the new default model

## Fixed

* Model name correctly updates from the template by default. However, you can still override the model name using the `--model` or `-m` option.

# 2024-02-10

## Added

* Add new OpenAI models' names (*-0125)

## Changed

* Pricing of OpenAI models

# 2023-11-06

## Changed

* The names of the OpenAI models have been modified following the changes made on
November 6th 2023.
