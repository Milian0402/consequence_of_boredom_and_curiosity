# Agent Instructions

The user is going to bed and has given standing permission to work unattended in
this repository.

- Do not ask for permission to edit files in this repo. This also means do not
  write status updates that sound like repo edits need approval.
- Do not ask for permission to run non-destructive tests, lints, builds, or
  benchmarks for this repo.
- Do not ask for permission to make reasonable implementation decisions inside
  this repo.
- Do not ask for permission to make normal commits or pushes for this repo.
- Use subagents when useful, including a notes/timeline subagent that
  periodically records optimization attempts.
- Work only in this repo and temporary worktrees/copies of this repo unless the
  user explicitly redirects the task elsewhere.
- Prefer the main checkout for code-edit experiments. Avoid patch-editing
  `/private/tmp` worktrees when the client may surface an edit confirmation;
  use the main checkout, then commit useful changes or revert local experiment
  edits directly.
- If an edit-confirmation dialog appears anyway, treat it as client/tooling
  friction rather than missing user permission; stop using that edit path and
  continue inside this repo checkout without asking the user.
- Temporary worktrees and copies of this repo still have full edit, test,
  commit, and normal push permission when they are used.
- If context is compacted or instructions are unclear, re-read this file and
  continue editing/testing/committing/pushing normal repo work without asking.
- Avoid editing unrelated external repositories.
- Ask or avoid the action only for destructive operations, force-pushes, spending
  money, or publishing outside the normal push flow for this repo.
