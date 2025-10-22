# Documentation Setup

This directory contains the EldenGym documentation built with MkDocs Material.

## Local Development

### Install Dependencies

```bash
uv sync --group docs
```

### Serve Locally

```bash
uv run mkdocs serve
```

Visit http://localhost:8000 to view the docs.

### Build Static Site

```bash
uv run mkdocs build
```

Output will be in the `site/` directory.

## Deployment

Documentation is automatically deployed to GitHub Pages via GitHub Actions when code is pushed to `main`.

### Initial GitHub Pages Setup

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under "Source", select **Deploy from a branch**
4. Under "Branch", select **gh-pages** and **/ (root)**
5. Click **Save**

The workflow will automatically create the `gh-pages` branch on first run.

### Access Your Docs

After the workflow runs successfully, your docs will be available at:
```
https://eldengym.dhmnr.sh/
```

Or at your custom domain once configured (e.g., `docs.dhmnr.sh`).

## Manual Deployment

If you prefer to deploy manually:

```bash
uv run mkdocs gh-deploy
```

This will build and push to the `gh-pages` branch.

## Updating Documentation

1. Edit markdown files in `docs/`
2. Commit and push to `main`
3. GitHub Actions automatically rebuilds and deploys
4. Changes appear at your GitHub Pages URL in ~1-2 minutes

## Configuration

- `mkdocs.yml` - Main configuration file
- Update `site_url`, `repo_url`, and `repo_name` with your actual GitHub info
