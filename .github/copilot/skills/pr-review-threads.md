# Skill: Handle unresolved PR review threads

## Purpose

Fetch all unresolved review threads for the current branch PR, classify each thread, then execute
approved actions in a batched and review-safe way.

## Inputs

- Repository and branch with an open PR.
- Access to GitHub GraphQL via `gh` (must be authenticated — run `gh auth status` to verify).

## Procedure

1. Fetch all PR review threads and paginate until `hasNextPage` is false.
1. Filter to unresolved threads only.
1. Classify each unresolved thread as one of:
   - code fix
   - reply-to-close
   - investigate
1. Report findings and ask user to pick one or more actions.
1. For approved code-fix actions, batch all edits together.
1. Always pause for user review before committing any changes.
1. After user review and approval:
   - commit and push changes
   - reply to each addressed thread with: `Fixed in <sha>`
   - resolve the thread

## Output format

- Section 1: unresolved thread table with URL, type, and recommendation.
- Section 2: proposed action batch.
- Section 3: post-push follow-up status with commit SHA.

## Guardrails

- Never stop at the first 100 threads; always paginate fully.
- Do not resolve threads until fixes are pushed and user approved.
- Keep code fixes batched to minimize review churn.
