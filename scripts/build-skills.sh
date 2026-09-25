#!/usr/bin/env bash
# Build the Claude.ai skill packages: dist/procedural-terrain-art.skill and dist/procedural-realism.skill.
#
# Each .skill is a zip with one top-level folder holding SKILL.md, scripts/, references/ (and assets/ or
# scenes/). Checks before packaging:
#   - frontmatter uses only Agent Skills spec fields (Claude Code-only keys fail the Claude.ai upload)
#   - description is under 1024 characters
#   - every references/*.md file SKILL.md points at exists
#   - scripts/dem.py is byte-identical in both skills (it is shared code)
# evals/ and caches are left out. Zips are byte-stable (sorted, no extra attributes, no dir entries).
#
# Usage: scripts/build-skills.sh [out-dir]   (default: dist/)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/dist}"; mkdir -p "$out"; out="$(cd "$out" && pwd)"

a="$root/skills/procedural-terrain-art/scripts/dem.py"; b="$root/skills/procedural-realism/scripts/dem.py"
cmp -s "$a" "$b" || { echo "scripts/dem.py differs between the two skills; copy one over the other." >&2; exit 1; }

forbidden='^(when_to_use|argument-hint|arguments|disable-model-invocation|user-invocable|disallowed-tools|model|effort|context|agent|background|hooks|paths|shell):'
for dir in "$root"/skills/*/; do
  name="$(basename "$dir")"; md="$dir/SKILL.md"
  fm="$(awk 'NR==1 && $0!="---"{exit} NR>1 && $0=="---"{exit} NR>1{print}' "$md")"
  [ -n "$fm" ] || { echo "$name: missing frontmatter" >&2; exit 1; }
  echo "$fm" | grep -qx "name: $name" || { echo "$name: frontmatter name must be $name" >&2; exit 1; }
  if echo "$fm" | grep -Eq "$forbidden"; then echo "$name: Claude Code-only frontmatter key" >&2; exit 1; fi
  len=$(echo "$fm" | sed -n 's/^description: *//p' | tr -d '\n' | wc -c)
  [ "$len" -gt 0 ] && [ "$len" -lt 1024 ] || { echo "$name: description is $len chars (1..1023)" >&2; exit 1; }
  for ref in $(grep -o 'references/[A-Za-z0-9_-]*\.md' "$md" | sort -u); do
    [ -f "$dir/$ref" ] || { echo "$name: SKILL.md points at missing $ref" >&2; exit 1; }
  done
  stage="$(mktemp -d)"; mkdir -p "$stage/$name"
  ( cd "$dir" && find . -type f ! -path './evals/*' ! -path '*/__pycache__/*' ! -name '*.pyc' ! -name '.DS_Store' \
      | sed 's#^\./##' | while read -r f; do mkdir -p "$stage/$name/$(dirname "$f")"; cp "$f" "$stage/$name/$f"; done )
  find "$stage" -exec touch -t 198001010000 {} +      # fixed mtimes: same input, same zip bytes
  rm -f "$out/$name.skill"
  ( cd "$stage" && find "$name" -type f | LC_ALL=C sort | TZ=UTC zip -X -D -q "$out/$name.skill" -@ )
  rm -rf "$stage"
  echo "built $out/$name.skill  ($(unzip -Z1 "$out/$name.skill" | wc -l | tr -d ' ') files, description $len chars)"
done
