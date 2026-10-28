# RegLens terminal demo: records a REAL run to docs/demo.gif
# Prereqs (from repo root): make install && make ingest, GEMINI_API_KEY exported
# Install VHS: brew install vhs  (or: go install github.com/charmbracelet/vhs@latest)
# Record:      vhs docs/demo.tape

Output docs/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 700
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

Hide
Type "source .venv/bin/activate && clear"
Enter
Show

Type "# Multi-part question: Planner splits it, Checker verifies every claim"
Enter
Type `reglens-ask "What is the Credit Risk-o-Meter, and what are the position limits for broad commodities?"`
Enter
Sleep 30s

Type "# Off-topic question: refused before any LLM call"
Enter
Type `reglens-ask "Who won the 2011 Cricket World Cup?"`
Enter
Sleep 12s
