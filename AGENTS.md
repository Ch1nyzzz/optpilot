# Repository Guidelines

## Project Structure & Module Organization

This repository is currently a lightweight research workspace centered on
[`related_papers.md`](./related_papers.md), which holds literature notes and project
positioning for a meta-optimizer research direction. Keep new top-level files focused
and obvious: use descriptive Markdown names such as `experiment_notes.md` or
`baseline_comparison.md`.

The repository is flat today. Do not scatter drafts across the root. If you add code,
data, or figures, introduce clear directories such as `scripts/`, `data/`, or
`figures/` and keep research notes in Markdown files.

Keep persistent project memory under `memory_bank/`:

- `memory_bank/project_goal.md` for goals and scope
- `memory_bank/progress.md` for milestone tracking
- `memory_bank/architecture.md` for architecture and structure decisions

Any architecture or repository-structure change must update
`memory_bank/architecture.md` in the same change.

## Build, Test, and Development Commands

There is no build system or automated test suite configured yet. Use lightweight
commands to inspect and validate changes:

- `rg --files` lists the current repository contents.
- `sed -n '1,120p' related_papers.md` spot-checks edited sections.
- `rg -n 'TODO|FIXME|XXX' .` finds unresolved placeholders before submission.
- `markdownlint AGENTS.md related_papers.md` is recommended if `markdownlint` is
  installed locally.

## Coding Style & Naming Conventions

Treat this repository as documentation-first. Use Markdown with ATX headings (`#`,
`##`), short paragraphs, and compact bullet lists. Keep filenames lowercase with
underscores, for example `related_papers.md`.

Preserve the language and tone of the file you are editing. For research content, name
papers and systems precisely on first mention and avoid informal shorthand unless it is
already defined.

## Testing Guidelines

Testing here means review and consistency checks rather than unit tests. Before opening
a change, verify heading structure, table formatting, and link syntax by reading the
rendered Markdown or linting it locally.

If you add code later, place tests beside that code or under a top-level `tests/`
directory and document the exact run command in this file.

## Commit & Pull Request Guidelines

There is no mature repo-local convention yet, but the available history uses short,
lowercase subjects such as `add init` and `readme`. Follow that pattern with concise,
imperative commit messages, for example `add paper summary` or `refine optimizer notes`.

Pull requests should state what changed, why it matters, and which sections or sources
were added or revised. Include screenshots only when Markdown rendering or figures are
part of the change.
