---
name: eval-gate
description: Run the eval harness against the tracked baseline and report the score deltas.
allowed-tools: Bash(python -m evals.runner:*), Bash(gh pr view:*)
---

Run `python -m evals.runner --compare` from the repository root. The working directory matters: the harness opens `./chroma_db` and reads `./data` by relative path, so running from anywhere else will fail or read the wrong store.

Confirm before running. A full run scores 40 items through generation plus three judges, takes roughly five to six minutes, and costs a few cents in API calls. Do not run it on a passing mention of evals.

Report the delta table exactly as printed. Do not judge whether a movement is significant. Per-category counts are small enough that a single judge flip moves a number by a tenth, so that call belongs to the user.

If the harness reports no baseline, `evals/baseline.json` is missing. Say so and stop. Writing a baseline is a deliberate decision made with `--baseline`, not a recovery step to take automatically.

After reporting the deltas, check whether the current branch has an open pull request and whether its body already records eval scores. The recorded form varies: look for three slash-separated numbers alongside an item count and an errored count, not for a particular heading. Some PR bodies label them, some carry them inside a verification section.

If no eval line is present, say so and hand the delta table over for the user to record. Do not draft the wording or edit the PR body. Recording scores in a PR body is part of merge-gate narration, which is the user's to author.
