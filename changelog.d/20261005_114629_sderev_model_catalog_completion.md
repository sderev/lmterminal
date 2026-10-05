### Added

* Register GPT-5.5, GPT-5.6 Sol/Terra/Luna, GPT-6 Sol and nine regular-text GPT-5 snapshots for Chat Completions, with prices and fixed aliases. `gpt-5.6` and `5.6` resolve to GPT-5.6 Sol; moving family aliases retain their newest releases.

### Changed

* Omit implicit temperature for GPT-5.5/5.6 and forward explicit sampling controls unchanged for provider validation; acceptance remains unverified. Omitted library temperature follows the model policy; explicit `None` omits the field.
* Reject explicit temperature `1` for known incompatible combinations instead of silently dropping it. Validate published reasoning efforts and reject known Responses-only GPT-5.5 Pro requests locally.
* Clarify that local uncached input estimates exclude cache reads and writes; actual input charges may be lower or higher.
