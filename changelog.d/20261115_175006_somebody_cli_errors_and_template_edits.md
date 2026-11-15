### Fixed
* Template add/edit validates editor saves before publication. Invalid saves and
  editor failures leave active templates untouched and retain changed drafts at
  a reported recovery path. Unsupported-field errors name bounded schema typos
  and list accepted fields without printing values.
