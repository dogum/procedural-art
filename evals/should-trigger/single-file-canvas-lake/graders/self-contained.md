---
type: llm
focus: { source: file, path: lake.html }
---

The file should be one self-contained HTML page that draws a landscape on a canvas with its own JavaScript.

PASS if it contains a canvas element and inline script that draws a sky, mountains and water procedurally, takes a seed from the address or a constant (no bare Math.random for the picture), sets a flag when it has finished drawing, and references no external script, stylesheet, font or image. FAIL if it loads a library or image, embeds a pre-made picture, or is a stub that draws only a gradient.
