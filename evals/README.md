# Evals

Behavioural tests for both skills, run with [`claude plugin eval`](https://code.claude.com/docs/en/plugin-evals):

- `should-trigger/`: requests a skill must pick up. Graders check that the right skill fired, that the image file exists, and (with regex checks over the run trace) that it used real coordinates, the right style, the water flags or a JSON scene; an LLM judge then looks at the PNG itself.
- `should-not-trigger/`: near misses (Blender settings, a Midjourney prompt, a QGIS how-to, a plain fact). Graders assert neither skill fired.

Each case runs with and without the plugin, so the report shows what the skills add.

```bash
claude plugin eval . --allow-tools Write Bash
claude plugin eval . --allow-tools Write Bash --case "x-header-rainier" --runs 1 --ablation none   # one case, cheap
```

The trigger cases render small drafts and need Python with `requirements.txt` installed and network access to AWS Terrain Tiles. Runs bill to your account. `evals/results/` is gitignored.

Each skill also carries skill-creator evals in `skills/<name>/evals/` (`evals.json` with expectations, `trigger_evals.json` with 20 should/shouldn't-trigger queries for description tuning).
