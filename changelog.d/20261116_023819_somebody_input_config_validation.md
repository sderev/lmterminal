Fixed
-----
* Reject empty or whitespace-only composed user input before reading credentials
  or sending a request. System instructions alone do not count as user input;
  nonblank template tasks remain valid. Omit blank system messages while keeping
  meaningful input whitespace intact.
