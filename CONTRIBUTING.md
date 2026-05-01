# Contributions

> [!NOTE]
> This repository uses an automated contributor system inspired by the [All Contributors][(allcontributors.org)](https://allcontributors.org/) specification to recognize all forms of contribution, including code, documentation, ideas, testing, and feedback.

To keep the contributors table organized across README files, this repository uses GitHub Actions to automate the process.

To add a new contributor to the table, refer to `.github/contributors.yml`.

Under the `contributors` section, add:
- Contributor name (`name`)
- GitHub username (`github`)
- Contribution role(s) (`roles`)

Once changes to the `.yml` file are saved and pushed, a `chore: update contributors table` commit will be made automatically by our `contributions-bot` across all available `README*.md` files.

# Commits

When contributions are made, we kindly ask you to use [Conventional Commits](https://www.conventionalcommits.org/) to keep the history readable and consistent.

When writing the commit message, try to explain **why** the change was made, not just what changed.

For example, instead of `fix: update threshold value`, prefer `fix: lower threshold to avoid losing small locules`.

---

## Format
```
<type>: <short description>
```
- Use **present tense**: "add feature" not "added feature"

---

## Types

| Type | When to use |
|------|-------------|
| `feat` | New functionality or parameter |
| `fix` | Bug or incorrect behavior |
| `docs` | Documentation only |
| `refactor` | Code reorganization, no behavior change |
| `chore` | Config, dependencies, maintenance |
| `style` | Formatting, indentation, no logic change |

---

## Examples

```
feat: add compare_clahe parameter to enhance_locule_contrast
feat: add CLI support for --no_morphology flag

fix: erosion_px not applied when mask_rois is empty
fix: color CSV missing when analyze_color is False

docs: add CLI page to User Guide
docs: update analyze_folder parameters for external class

refactor: move symmetry functions to separate module

chore: update mkdocs.yml nav structure
chore: include opencv dependency

style: fix indentation in mask.py
```
