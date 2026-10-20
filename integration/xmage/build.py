"""Build the XMage distribution MageZero plays in, from an official XMage release.

    python integration/xmage/build.py                    # the release pinned in xmage-release.txt
    python integration/xmage/build.py --release latest   # the newest official release
    python integration/xmage/build.py --check            # compile only (CI), nothing installed

Steps: download the release's mage-full ZIP from github.com/magefree/mage (cached in
.mz_cache/), copy its server (lib/, plugins/, config/) into the destination, compile
the adapter (src/, src-ma/) against those jars into mz/magezero-adapter.jar, and
write the launchers, the `mz play` server configuration and the decks.

XMage itself is never patched or rebuilt: the adapter only extends its public player
API. The optional engine fixes in patches/ are applied to the release's own copy of
each file (downloaded for the release's tag) and compiled into mz/engine-fixes.jar,
which comes first on the classpath. A fix that no longer applies or compiles against
a newer release is skipped with a warning; games still run, without that fix.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PIN = HERE / "xmage-release.txt"
API = "https://api.github.com/repos/magefree/mage/releases"
RAW = "https://raw.githubusercontent.com/magefree/mage"
ADAPTER_SOURCES = ("src", "src-ma")
SERVER = "xmage/mage-server/"  # the server inside the release ZIP
COPIED = ("lib", "plugins", "config")  # replaced on every build; decks/, db/ and logs are kept
MAIN = "org.mage.magezero.PolicyGames"
POLICY_PLAYER = "mage.player.ai.policy.PolicyPlayer"
# `mz play`: the type stock XMage clients offer as an opponent seat.
POLICY_PLAYER_TYPE = "Computer - monte carlo"


def write_text(path: Path, text: str) -> None:
    """Write `text` exactly (no newline translation), so launchers keep their line endings on every OS."""
    with path.open("w", encoding="utf-8", newline="") as out:
        out.write(text)


def log(message: str) -> None:
    print(f"[xmage] {message}", flush=True)


def github(url: str):
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def resolve_release(name: str) -> tuple[str, str, str]:
    """(tag, asset name, download URL) of an official release: a tag, or "latest"."""
    release = github(f"{API}/latest" if name == "latest" else f"{API}/tags/{name}")
    assets = [a for a in release["assets"] if re.fullmatch(r"mage-full_.*\.zip", a["name"])]
    if len(assets) != 1:
        raise SystemExit(f"release {release['tag_name']} has no single mage-full ZIP: {[a['name'] for a in assets]}")
    return release["tag_name"], assets[0]["name"], assets[0]["browser_download_url"]


def download(url: str, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, shown = 0, 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total and done - shown >= total // 10:
                shown = done
                log(f"  {done * 100 // total}%")
    partial.replace(target)
    return target


def release_version(archive: Path) -> str:
    """The XMage version of a release ZIP, from its core jar (mage-1.4.62.jar -> 1.4.62)."""
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            match = re.fullmatch(re.escape(SERVER) + r"lib/mage-(\d+(?:\.\d+)+)\.jar", name)
            if match:
                return match.group(1)
    raise SystemExit(f"{archive.name} has no {SERVER}lib/mage-<version>.jar")


def extract_server(archive: Path, dest: Path) -> None:
    for folder in COPIED:
        shutil.rmtree(dest / folder, ignore_errors=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if not info.filename.startswith(SERVER) or info.is_dir():
                continue
            relative = Path(info.filename[len(SERVER) :])
            if relative.parts[0] not in COPIED:
                continue
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)


def javac() -> str:
    home = os.environ.get("JAVA_HOME")
    candidates = [Path(home) / "bin" / ("javac.exe" if os.name == "nt" else "javac")] if home else []
    found = shutil.which("javac")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise SystemExit("javac not found: install a JDK (21 or newer) and put it on PATH or set JAVA_HOME")


def compile_java(sources: list[Path], classpath: list[Path], out: Path) -> subprocess.CompletedProcess:
    out.mkdir(parents=True, exist_ok=True)
    # An argument file keeps the command short on Windows.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as listing:
        listing.write("\n".join(f'"{p.as_posix()}"' for p in sources))
    try:
        command = [javac(), "-nowarn", "-encoding", "UTF-8", "-cp", os.pathsep.join(map(str, classpath)),
                   "-d", str(out), f"@{listing.name}"]  # fmt: skip
        return subprocess.run(command, capture_output=True, text=True)
    finally:
        os.unlink(listing.name)


def write_jar(classes: Path, jar: Path, extra: dict[str, bytes] | None = None) -> None:
    jar.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(jar, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\r\nCreated-By: MageZero build.py\r\n\r\n")
        for path in sorted(classes.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(classes).as_posix())
        for name, data in (extra or {}).items():
            zf.writestr(name, data)


def xmage_classpath(dest: Path) -> list[Path]:
    return [dest / "lib" / "*", dest / "plugins" / "*"]


def build_adapter(dest: Path, work: Path) -> str:
    sources = sorted(p for folder in ADAPTER_SOURCES for p in (HERE / folder).rglob("*.java"))
    result = compile_java(sources, xmage_classpath(dest), work / "adapter")
    if result.returncode != 0:
        raise SystemExit(f"the adapter does not compile against this XMage release:\n{result.stdout}{result.stderr}")
    write_jar(work / "adapter", dest / "mz" / "magezero-adapter.jar")
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.relative_to(HERE).as_posix().encode() + b"\0" + path.read_bytes())
    log(f"compiled the adapter ({len(sources)} sources) into mz/magezero-adapter.jar")
    return digest.hexdigest()


def build_engine_fixes(dest: Path, tag: str, work: Path) -> dict[str, str]:
    """Apply each patch to this release's copy of its file; returns {patch: applied | skipped: reason}."""
    status: dict[str, str] = {}
    sources = []
    tree = work / "fixes-src"
    for patch in sorted((HERE / "patches").glob("*.patch")):
        targets = re.findall(r"^\+\+\+ b/(\S+)", patch.read_text(encoding="utf-8"), re.MULTILINE)
        try:
            if shutil.which("git") is None:
                raise RuntimeError("git is not installed")
            for target in targets:
                file = tree / target
                if not file.exists():
                    with urllib.request.urlopen(f"{RAW}/{tag}/{target}", timeout=60) as response:
                        file.parent.mkdir(parents=True, exist_ok=True)
                        file.write_bytes(response.read())
            check = subprocess.run(["git", "apply", "--check", "--ignore-whitespace", str(patch)], cwd=tree,
                                   capture_output=True, text=True)  # fmt: skip
            if check.returncode != 0:
                raise RuntimeError(f"does not apply: {check.stderr.strip().splitlines()[0]}")
            subprocess.run(["git", "apply", "--ignore-whitespace", str(patch)], cwd=tree, check=True)
            sources += [tree / t for t in targets]
            status[patch.name] = "applied"
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            status[patch.name] = f"skipped: {error}"
            log(f"WARNING: engine fix {patch.name} skipped for {tag} ({error}); games run without it")
    jar = dest / "mz" / "engine-fixes.jar"
    jar.unlink(missing_ok=True)
    if sources:
        result = compile_java(sources, xmage_classpath(dest), work / "fixes")
        if result.returncode != 0:
            for name in status:
                if status[name] == "applied":
                    status[name] = "skipped: does not compile against this release"
            log(f"WARNING: engine fixes skipped, they do not compile against {tag}:\n{result.stdout}{result.stderr}")
        else:
            write_jar(work / "fixes", jar)
            log(f"engine fixes in mz/engine-fixes.jar: {', '.join(n for n, s in status.items() if s == 'applied')}")
    return status


def write_launchers(dest: Path) -> None:
    """mz-policy.bat / .sh: PolicyGames with the adapter and fixes ahead of XMage's jars."""
    write_text(
        dest / "mz-policy.bat",
        "@echo off\r\n"
        "setlocal\r\n"
        'if "%~1"=="" exit /b 2\r\n'
        'if "%MZ_HEAP%"=="" set MZ_HEAP=8G\r\n'
        'pushd "%~dp0"\r\n'
        f'java -Xmx%MZ_HEAP% -Dh2.bindAddress=127.0.0.1 -cp "mz/*;lib/*;plugins/*" {MAIN} %*\r\n'
        "set POLICY_EXIT=%ERRORLEVEL%\r\n"
        "popd\r\n"
        "exit /b %POLICY_EXIT%\r\n",
    )
    script = dest / "mz-policy.sh"
    write_text(
        script,
        "#!/bin/sh\n"
        '[ -n "$1" ] || exit 2\n'
        'cd "$(dirname "$0")" || exit 2\n'
        f'exec java -Xmx"${{MZ_HEAP:-8G}}" -Dh2.bindAddress=127.0.0.1 -cp "mz/*:lib/*:plugins/*" {MAIN} "$@"\n',
    )
    script.chmod(0o755)


def write_server_config(dest: Path) -> None:
    """config/magezero-server.xml: the release's own config.xml, local only, with the policy as a player type."""
    text = (dest / "config" / "config.xml").read_text(encoding="utf-8")
    for attribute, value in {
        "serverAddress": "127.0.0.1",
        "serverName": "magezero-local",
        "secondaryBindPort": "17179",
        "maxSecondsIdle": "1800",
    }.items():
        text, count = re.subn(rf'(\s{attribute}=")[^"]*(")', rf"\g<1>{value}\g<2>", text, count=1)
        if count != 1:
            raise SystemExit(f"config.xml of this release has no {attribute} attribute")
    # The policy plays the "Computer - monte carlo" seat. Its class is on the server's
    # classpath (mz/), so the jar attribute only has to name a plugin that exists.
    plugin = re.search(r'<playerType name="Computer - mad" jar="([^"]+)"', text)
    if plugin is None:
        raise SystemExit('config.xml of this release has no "Computer - mad" player type')
    text = re.sub(rf'\s*<playerType name="{re.escape(POLICY_PLAYER_TYPE)}"[^>]*/>', "", text)
    entry = f'<playerType name="{POLICY_PLAYER_TYPE}" jar="{plugin.group(1)}" className="{POLICY_PLAYER}"/>'
    text, count = re.subn(r"(\n(\s*)</playerTypes>)", rf"\n\g<2>    {entry}\1", text, count=1)
    if count != 1:
        raise SystemExit("config.xml of this release has no </playerTypes>")
    header = (
        "<!-- MageZero human-play server (mz play), written by integration/xmage/build.py from this\n"
        '     release\'s config.xml. Local only. "Computer - monte carlo" is the policy (PolicyPlayer);\n'
        "     its decisions come from the service named by -Dmagezero.policy.endpoint. -->\n"
    )
    text = re.sub(r"(<\?xml[^>]*\?>\s*)", rf"\g<1>{header}", text, count=1)
    write_text(dest / "config" / "magezero-server.xml", text)


def install_decks(dest: Path) -> None:
    (dest / "decks").mkdir(parents=True, exist_ok=True)
    for deck in sorted((REPO / "decks").glob("*.dck")):
        shutil.copyfile(deck, dest / "decks" / deck.name)


def build(release: str, dest: Path, cache: Path, check: bool, replace: bool) -> dict:
    tag, asset, url = resolve_release(release)
    log(f"official XMage release {tag} ({asset})")
    archive = download(url, cache / tag / asset)
    version = release_version(archive)
    manifest_path = dest / "mz" / "build.json"
    if not check and (dest / "lib").exists() and not manifest_path.exists() and not replace:
        raise SystemExit(
            f"{dest} holds an XMage distribution this script did not build; move it away or pass --replace"
        )
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    with tempfile.TemporaryDirectory(prefix="mz-xmage-") as temp:
        work = Path(temp)
        target = work / "dist" if check else dest
        target.mkdir(parents=True, exist_ok=True)
        extract_server(archive, target)
        adapter = build_adapter(target, work)
        fixes = build_engine_fixes(target, tag, work)
        if check:
            log(f"the adapter compiles against XMage {version} ({tag})")
            return {"release": tag, "xmage_version": version, "engine_fixes": fixes}
        if previous.get("xmage_version") != version:
            # The card database belongs to one XMage version; XMage rebuilds it on first start.
            shutil.rmtree(dest / "db", ignore_errors=True)
        # Compiled by `mz web` against the previous jars.
        shutil.rmtree(dest / "mz" / "web-classes", ignore_errors=True)
        write_launchers(dest)
        write_server_config(dest)
        install_decks(dest)
    manifest = {
        "release": tag,
        "asset": asset,
        "xmage_version": version,
        "adapter_sources_sha256": adapter,
        "engine_fixes": fixes,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    launcher = dest / ("mz-policy.bat" if os.name == "nt" else "mz-policy.sh")
    log(f"built {dest} on XMage {version}; engine launcher: {launcher}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--release",
        default=PIN.read_text(encoding="utf-8").strip(),
        help='official release tag (e.g. xmage_1.4.62V1) or "latest"; default: xmage-release.txt',
    )
    parser.add_argument("--dest", type=Path, default=REPO / "xmage", help="distribution folder (default: xmage/)")
    parser.add_argument("--cache", type=Path, default=REPO / ".mz_cache" / "xmage", help="release download cache")
    parser.add_argument("--check", action="store_true", help="only check that the adapter compiles; install nothing")
    parser.add_argument("--replace", action="store_true", help="replace a distribution this script did not build")
    args = parser.parse_args()
    build(args.release, args.dest.resolve(), args.cache.resolve(), args.check, args.replace)


if __name__ == "__main__":
    sys.exit(main())
