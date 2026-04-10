import re
import yaml
from pathlib import Path

with open(".github/contributors.yml", "r") as f:
    config = yaml.safe_load(f)

roles_map = config["roles_map"]
contributors = config["contributors"]
translations = config["translations"]

def get_lang(path):
    return "es" if "_es" in path.lower() else "en"

def build_table(path):
    t = translations[get_lang(path)]
    rows = []
    for c in contributors:
        emojis = " ".join(roles_map.get(r, f"Unknown role:({r})") for r in c["roles"])
        if "github" in c:
            avatar = f'<img src="https://github.com/{c["github"]}.png" width="44" height="44" valign="middle">&nbsp;'
            name = f'[{avatar}{c["name"]}](https://github.com/{c["github"]})'
        else:
            name = c["name"]
        rows.append(f"| {name} | {emojis} |")
    return "\n".join([
        f"| {t['contributor']} | {t['role']} |",
        "|-------------|------|",
        *rows,
    ])

def update_readme(path):
    if not Path(path).exists():
        return
    content = Path(path).read_text(encoding="utf-8")
    table = build_table(path)
    updated = re.sub(
        r"<!-- CONTRIBUTORS-START -->[\s\S]*?<!-- CONTRIBUTORS-END -->",
        f"<!-- CONTRIBUTORS-START -->\n{table}\n<!-- CONTRIBUTORS-END -->",
        content
    )
    Path(path).write_text(updated, encoding="utf-8")
    print(f"Successfully updated: {path}")

update_readme("README.md")
update_readme("README_ES.md")
