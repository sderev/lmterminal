### Changed

* Live API tests require both `--run-live` and `--live-model MODEL` and run one
  selected model. Ordinary tests retain their offline guards, and `gate` excludes
  live tests across every Python version.
