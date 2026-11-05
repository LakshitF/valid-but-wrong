# Visual explainer

A single-page, scroll-driven explainer for *Valid but Wrong: Silent Enum
Substitution in Tool Calls, and Why Schema Validation Cannot See It*.

## View it

From the repository root:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/site/>.

The "Paper (PDF)" links point to `../paper/main.pdf`, so serve from the
repository root. Opening `site/index.html` directly also works in a browser
that permits local `file://` pages.

## Files

| File | Purpose |
|---|---|
| `index.html` | All content, diagrams and section structure |
| `styles.css` | Theme, layout, diagrams, charts, responsive rules, reduced motion |
| `app.js` | Progress bar, scrollspy, reveal animations, three interactive controls, chart highlighting, copy citation |

No build step and no JavaScript dependencies. The only external request is
Google Fonts; the page falls back to system fonts offline.

## Notes

- Every number comes from `paper/main.md` (and the released result JSONs).
  The page is an explainer, not a substitute for the paper.
- All diagrams are drawn directly in HTML/CSS/SVG. None of the PDF figures in
  `figures/` are copied in; the charts are redrawn for the dark theme.
- The three interactive arms are:
  - the abstention / permission-to-ask prompt toggle;
  - the validation-retry wording toggle (with or without the allowed values);
  - the hosted provider's enum-enforced vs enum-removed schema toggle.
- Reveal animations and controls are progressive enhancement. The page still
  reads with JavaScript disabled.
- The layout is tested at desktop and phone widths with no horizontal page
  overflow. Wide charts and tables scroll inside their own containers.
