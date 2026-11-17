# DriveHierarchy project page

A buildless GitHub Pages site for the accepted NeurIPS 2026 Evaluations & Datasets paper. Everything needed at runtime is local HTML, CSS, JavaScript, WebP/SVG images. No external fonts, analytics, UI libraries, build system, or hosted model calls are used.

## Preview and publish

From the DriveHierarchy repository:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory project-page
```

Open http://127.0.0.1:8000. Directly opening index.html also works; copying the citation has a manual-selection fallback.

The existing GitHub Pages workflow remains **manual-only**: Settings → Pages → Source → GitHub Actions, then run **Deploy DriveHierarchy project page**.

The workflow stages only index.html, style.css, script.js, WebP/SVG images and .nojekyll. It excludes the legacy local paper PDF, these notes and preparation tools.

Both Paper buttons and the BibTeX URL point to the author-provided NeurIPS page: https://neurips.cc/virtual/2026/loc/paris/poster/139730. The legacy PDF remains local and is excluded from publication.

## Design

The palette develops Color Hunt’s [blue/pink combination](https://colorhunt.co/palettes/blue-pink): ink blue `#1B3C73`, steel blue `#40679E`, blush `#FFCAD4`, and rose `#FF407D`. A warm near-white background and darker text variants keep the page readable. A four-layer line symbol replaces the letter monogram in the navigation, hero and favicon.

The cover is a complete viewport composition. Its minimum height is the small viewport height minus the sticky header; text can expand naturally on short screens or at increased zoom. Research images and headline counts start in the following overview section. Three resource buttons, linked authors and a four-rank navigation strip complete the cover.

The original background uses four suspended contour planes, nested paths, a connecting spine and moving signals. Blue progresses through muted violet to rose as the signal climbs from R1 to R4. Gentle layer movement and expanding node rings make the hierarchy visible immediately. A fine pointer leaves a thin blue/pink tail capped at 76 px and 170 ms. Crossing a layer face triggers one 620 ms lift-and-settle response on that individual layer. Stationary SVG wrappers provide stable geometric hit testing. There is no pointer halo or ripple. Feedback stops on pointer exit, scrolling, tab blur, hidden pages or paused/reduced motion; touch and forced-color modes omit pointer decoration. Effects pause through the navigation control, when the tab is hidden, and with a reduced-motion preference.

The document uses native CSS `scroll-snap-type: y mandatory` with section-start targets, normal snap stops and a sticky-header offset for a firmer section landing. Long sections remain freely scrollable; wheel and touch gestures are handled by the browser. Reduced-motion mode disables snapping and smooth scrolling.

Arial/Helvetica, normal tracking, bold section headings up to 64 px, and a 1,340 px content width support the promotional presentation. Large model scores and the 0.902 → 8.021 SFT result lead the lower half. Full model results are in a searchable disclosure. Video playback, its assets and its deployment step have been removed; original recordings remain in the author's Videos directory.

## Evidence and data

The authoritative source files are [main.tex](../../DriveHierarchyNeurIPS/main.tex), [QAs.tex](../../DriveHierarchyNeurIPS/QAs.tex) and [R4_example.tex](../../DriveHierarchyNeurIPS/R4_example.tex).

- Identity: main.tex lines 100 and 128–137; NeurIPS final Evaluations & Datasets configuration at line 36, supported by the author's acceptance confirmation.
- Full corpus: 76,798 QA pairs / 84,279 frames. All reported R1–R3 results use the refined **14,000 QA / 31,945-frame** subset (main.tex line 226).
- R1: 8 tasks; R2: 4; R3: 2; R4: 100 scenarios in 10 families.
- Headline results: Tables 1–2, 15 models, transcribed into script.js. Open-loop and closed-loop scores use different protocols; deterministic single-run results do not support strong claims about small differences.
- Open-loop weights are frozen after calibration; rank budgets are 3:3:4. R4 combines route completion, safety and efficiency multiplicatively.
- R2 example: all six synchronized cameras from Fig/example_2_1. Ground truth **10 unique vehicles**, from QAs.tex lines 354–367.
- R3 example: original A–E frames from Fig/example_3_2. Ground truth **E → D → B → A → C**, from QAs.tex lines 477–490. The interactive check uses pairwise-order consistency.
- Transfer example: Qwen3-VL-8B-Instruct, baseline 0.902 → joint weak-set SFT 8.021. No R4 scenario was a supervision target. This is one base-model study, not a universal gain claim.
- The transfer plot displays 0–10 with an explicit axis; the complete R4 metric is bounded by 0–100.
- The road layout is real-world; evaluation is in simulation. The site does not assert deployment readiness.

Original figure colors and scientific content are preserved. No model responses, trajectory predictions, scenario scores, or experimental footage were synthesized.

## Verification

HTML references, SVG reference IDs, ARIA targets, CSS/JavaScript parsing and core text contrast are checked. The current revision checks the cover/overview boundary, four layer symbols, Chengkai Xu's homepage link, Jian Sun's plain-text attribution, removal of all video references and video packaging, and native scroll-snap settings.

Offline DOM checks cover sorting/search, dataset scopes, capability navigation, benchmark answers, scenario and profile switching, Paper destinations, motion preferences, the mobile menu and citation fallback. These are logic checks; native scrolling and desktop/mobile rendering still require a browser review. The browser preview infrastructure was unavailable in this environment.
