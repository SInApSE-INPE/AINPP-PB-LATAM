# Publishing Documentation

This project uses `MkDocs` with the `Material` theme and publishes the generated static site to GitHub Pages through GitHub Actions.

## Local Preview

Install the project with the docs extras and start the local server:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[docs]
uv run mkdocs serve
```

The local preview is available at `http://127.0.0.1:8000`.

## Production Build

To validate the generated static site before publishing:

```bash
uv run mkdocs build --strict
```

The output is written to the `site/` directory.

## GitHub Pages Deployment

The repository includes a workflow at `.github/workflows/docs.yml` that:

1. installs the package with the `docs` extras,
2. builds the site with `mkdocs build --strict`,
3. uploads the generated `site/` artifact,
4. deploys the artifact to GitHub Pages.

## One-Time GitHub Setup

In the GitHub repository settings:

1. Open `Settings > Pages`.
2. In `Build and deployment`, select `GitHub Actions` as the source.
3. Keep the default branch as `main`.

After that, each push to `main` updates the published documentation automatically.

## Updating the Content

- Edit Markdown pages inside `docs/`.
- Keep API reference pages in sync with package modules and docstrings.
- Update `mkdocs.yml` when adding new sections to the navigation.
