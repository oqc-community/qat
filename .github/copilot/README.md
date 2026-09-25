# Copilot shared configuration

This directory contains repository-shared Copilot workflow assets for both JetBrains (PyCharm
plugin) and VS Code users.

## Layout

- `skills/`: reusable workflow playbooks.

## Cross-editor guidance

- Keep `AGENTS.md` (repository root) focused on always-on policy and coding conventions. Claude Code
  reads it, and so do Copilot code review on GitHub.com, the Copilot cloud agent and CLI, and
  Copilot Chat in VS Code and JetBrains.
- Copilot code review run inside VS Code or Visual Studio does not read `AGENTS.md`. For local
  reviews, use the team's `review-pr` agent skill instead.
- Keep multi-step workflows in `skills/` and reference them from `AGENTS.md`.
- Do not duplicate workflow logic in editor-specific settings files.

## How to invoke a skill

Skills are Copilot instruction fragments, not executable scripts. To use one, tell Copilot:

> "Follow the instructions in `.github/copilot/skills/<skill-name>.md` to …"

Copilot will read the file and follow the procedure described inside it.

## Skill index

- `skills/pr-review-threads.md` — triage and resolve unresolved PR review threads
- `skills/jira-ticket.md` — create or update a Jira ticket in ADF format
- `skills/pr-description.md` — generate a PR title and body safe for `gh`
