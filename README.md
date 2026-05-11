# paper-site-template

Reusable static template for technical paper websites.

## Design goals

- Berkeley Mono everywhere, including body copy and code
- White background, minimal borders, no gradients
- Centered paper-style hero with compact link pills
- Figure-first layout with math support
- KaTeX support for inline and display math
- No framework or build step
- Works on GitHub Pages

## Files

- `index.html`: page shell
- `content.js`: all paper-specific content lives here
- `script.js`: renderer for the hero, sections, links, and metadata
- `styles.css`: minimal research-oriented styling
- `assets/figures/`: replace placeholder figures with your own
- `.github/workflows/deploy.yml`: GitHub Pages deployment workflow

## Reusing for a new project

1. Copy this directory into a new repo.
2. Edit `content.js`.
3. Replace the placeholder figures in `assets/figures/`.
4. Update any paper/code/arXiv links.
5. Push to a public GitHub repo and enable Pages with `GitHub Actions`.

## Local preview

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Notes

- Inline math works with `$...$`.
- Display math works with `$$...$$`.
- Content strings are trusted HTML, so you can use tags like `<code>` and `<em>` inside `content.js`.
- If you want a different body font later, change the `@font-face` and `body` font stack in `styles.css`.
