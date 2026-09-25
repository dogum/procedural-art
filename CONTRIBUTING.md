# Contributing

Thanks for helping. The bar: the skills should make better pictures, stay honest about what code can't do yet, and never reach for an image model.

## Layout

- `skills/procedural-terrain-art/` and `skills/procedural-realism/`: each has a `SKILL.md` (in context whenever the skill fires, so keep it lean), `references/` (read on demand), and `scripts/`.
- `skills/*/scripts/dem.py` is shared code and must stay byte-identical in both skills. Edit one, copy it over the other.
- `evals/` holds `claude plugin eval` cases; `skills/*/evals/` holds skill-creator evals. `docs/` is the GitHub Pages site. `scripts/` holds the build and the terrain regression sheet.

## Before you open a PR

```bash
pip install -r requirements.txt
scripts/build-skills.sh                        # both .skill files still build and pass the checks
python scripts/regression_terrain.py --rerender   # if you touched the terrain engine: 7 landforms × 6 styles, one sheet
claude plugin validate .                       # manifests are valid
```

Look at what you rendered before you commit. Most improvements in this repo came from looking at a small draft, not from reasoning about the code. Include a before/after image in the PR for any change that affects pixels.

If you changed when a skill triggers (its `description`), run the evals and paste the summary in the PR:

```bash
claude plugin eval . --allow-tools Write Bash
```

## Rules of thumb

- **No image models, no downloaded imagery or textures.** Elevation tiles are the only external data.
- **Keep defaults stable.** A new feature is a flag; the default render of the documented workflows should not change unless that's the point of the PR.
- **Resumable or it isn't done.** Anything that can take minutes must survive a killed shell (row bands, sample accumulators, per-frame files).
- **No hardcoded paths.** Scripts read from their own folder and write to the working directory.
- **Frontmatter stays spec-only** (`name`, `description`, `license`, `compatibility`, `metadata`, `allowed-tools`). `scripts/build-skills.sh` refuses Claude Code-only keys.

## Releasing

1. Bump `version` in `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json` and `metadata.version` in both `SKILL.md` files.
2. Add a section to `CHANGELOG.md`.
3. Tag `vX.Y.Z` and push the tag. The release workflow builds both `.skill` files and creates the GitHub Release with the changelog section as notes.
