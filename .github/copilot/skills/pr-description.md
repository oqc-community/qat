# Skill: Build PR title and description

## Purpose

Generate PR title and body that follow repository policy and are safe to submit via `gh` without
shell-quoting issues.

## Inputs

- Jira ticket key.
- Short change summary.
- Motivation/context.
- Testing performed.
- Related issue links.
- Additional notes.

## Procedure

1. Generate PR title as `<jira-ticket>: <short summary>`.
1. Generate body with these required section titles:
   - `Summary`
   - `Motivation`
   - `Testing`
   - `Related Issues`
   - `Notes`
1. If using `gh pr create` or `gh pr edit`, write body markdown to a file and pass it using
   `--body-file`.
1. Return final title and body preview.

## Output format

- Title line.
- Five-section markdown body.
- Safe `gh` invocation pattern using `--body-file`.

## Guardrails

- Keep title concise and informative.
- Preserve literal section titles exactly.
- Never pass markdown with backticks directly to `--body`.
