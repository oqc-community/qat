# Copilot shared configuration

This directory contains repository-shared Copilot workflow assets for both JetBrains (PyCharm
plugin) and VS Code users.

## Layout

- `skills/`: reusable workflow playbooks.

## Cross-editor guidance

- Keep `.github/copilot-instructions.md` focused on always-on policy and coding conventions.
- Keep multi-step workflows in `skills/` and reference them from `.github/copilot-instructions.md`.
- Do not duplicate workflow logic in editor-specific settings files.

## How to invoke a skill

Skills are Copilot instruction fragments, not executable scripts. To use one, tell Copilot:

> "Follow the instructions in `.github/copilot/skills/<skill-name>.md` to …"

Copilot will read the file and follow the procedure described inside it.

## Skill index

- `skills/pr-review-threads.md` — triage and resolve unresolved PR review threads
- `skills/jira-ticket.md` — create or update a Jira ticket in ADF format
- `skills/pr-description.md` — generate a PR title and body safe for `gh`
