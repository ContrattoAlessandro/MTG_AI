"""Install the source overlay into an explicitly supplied XMage checkout.

It adds the policy classes and applies the small engine fixes in patches/
(each a git diff against the pinned fork; already-applied patches are skipped).
Run the normal engine build afterward; the separate PolicyGames entrypoint
does not invoke the old generator. Pin matches the fork reviewed for this change.
"""

import argparse
from pathlib import Path
import shutil
import subprocess

EXPECTED_COMMIT = "cb7e9c6fe2dcbe2f650a77dd98a87771e730f22f"


def install(root):
    root = Path(root).resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"adapter was reviewed against {EXPECTED_COMMIT}; checkout differs ({commit})")
    source = Path(__file__).resolve().parent / "src"
    for file in source.rglob("*.java"):
        relative = file.relative_to(source)
        module = (root / "Mage.MageZero" if relative.parts[0] == "org"
                  else root / "Mage.Server.Plugins" / "Mage.Player.AI")
        destination = module / "src" / "main" / "java" / relative
        if not destination.resolve().is_relative_to(root):
            raise ValueError("source target escaped the explicitly supplied checkout")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, destination)
    # Classes extending XMage's Mad AI (Mage.Player.AI.MA keeps a non-Maven src/ layout).
    for file in (source.parent / "src-ma").rglob("*.java"):
        destination = root / "Mage.Server.Plugins" / "Mage.Player.AI.MA" / "src" / file.relative_to(source.parent / "src-ma")
        if not destination.resolve().is_relative_to(root):
            raise ValueError("source target escaped the explicitly supplied checkout")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, destination)
    for patch in sorted((source.parent / "patches").glob("*.patch")):
        applied = subprocess.run(["git", "-C", str(root), "apply", "--reverse", "--check", "--ignore-whitespace", str(patch)],
                                 capture_output=True).returncode == 0
        if not applied:
            subprocess.run(["git", "-C", str(root), "apply", "--ignore-whitespace", str(patch)], check=True)
            print(f"Applied {patch.name}")
    for filename in ("mz-policy.bat", "mz-policy.sh"):
        destination = root / "Mage.MageZero" / "release" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.parent / filename, destination)
    print(f"Installed policy source overlay into {root}; rebuild the engine distribution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout")
    install(parser.parse_args().checkout)
