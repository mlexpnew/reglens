# RegLens terminal demo: records a REAL run to docs/demo.gif
# Prereqs (from repo root): pip install -e ".[dev,hf,agents]", make ingest, GEMINI_API_KEY exported
# Install VHS: brew install vhs  (or: go install github.com/charmbracelet/vhs@latest)
# Record:      vhs docs/demo.tape
# Timing is real: each step waits until the answer appears, then pauses so viewers can read it.
# Prompt is reset to "$ " (zsh themes do not work in bash); only known noisy warnings are filtered, real errors still show.

Output docs/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 700
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 20ms
Set WaitTimeout 90s

Hide
Type `export PS1="$ " PROMPT_COMMAND= TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error; source .venv/bin/activate; export PS1="$ "; reglens-ask() { command reglens-ask "$@" 2> >(grep -v -E "Redirects are currently|HF_TOKEN|unauthenticated requests|Loading weights" >&2); }; clear`
Enter
Show

Type "# Multi-part question: Planner splits it, Checker verifies every claim"
Enter
Type `reglens-ask "What is the Credit Risk-o-Meter, and what are the position limits for broad commodities?"`
Enter
Wait+Screen /latency/
Sleep 5s

Hide
Type "clear"
Enter
Show

Type "# Off-topic question: refused before any LLM call"
Enter
Type `reglens-ask "Who won the 2011 Cricket World Cup?"`
Enter
Wait+Screen /latency/
Sleep 4s
