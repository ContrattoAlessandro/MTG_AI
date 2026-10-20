"""integration/xmage/build.py: the parts that adapt to each official XMage release."""

import importlib.util
import zipfile
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[1] / "integration" / "xmage" / "build.py"
spec = importlib.util.spec_from_file_location("xmage_build", BUILD)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

CONFIG = """<?xml version="1.0" encoding="UTF-8"?>
<!-- RELEASE CONFIG for public -->
<config>
    <server serverAddress="0.0.0.0"
            serverName="mage-server"
            port="17171"
            secondaryBindPort="-1"
            maxSecondsIdle="600"
    />
    <playerTypes>
        <playerType name="Human" jar="mage-player-human-1.4.99.jar" className="mage.player.human.HumanPlayer"/>
        <playerType name="Computer - mad" jar="mage-player-ai-mad-1.4.99.jar" className="mage.player.ai.Proxy"/>
        <playerType name="Computer - monte carlo" jar="mage-player-ai-mcts-1.4.99.jar" className="mage.player.ai.M"/>
    </playerTypes>
</config>
"""


def test_server_config_is_the_release_config_with_the_policy_seat(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.xml").write_text(CONFIG, encoding="utf-8")
    build.write_server_config(tmp_path)
    text = (tmp_path / "config" / "magezero-server.xml").read_text(encoding="utf-8")
    assert 'serverAddress="127.0.0.1"' in text and 'serverName="magezero-local"' in text
    assert 'secondaryBindPort="17179"' in text and 'maxSecondsIdle="1800"' in text and 'port="17171"' in text
    # The release's own monte carlo entry is replaced by the policy, on a plugin jar of this release.
    assert text.count('name="Computer - monte carlo"') == 1
    assert (
        '<playerType name="Computer - monte carlo" jar="mage-player-ai-mad-1.4.99.jar" '
        'className="mage.player.ai.policy.PolicyPlayer"/>' in text
    )
    assert text.index("PolicyPlayer") < text.index("</playerTypes>")
    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<!-- MageZero human-play server')


def test_server_config_refuses_a_release_it_cannot_adapt(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.xml").write_text(CONFIG.replace("Computer - mad", "Computer - other"))
    with pytest.raises(SystemExit, match="Computer - mad"):
        build.write_server_config(tmp_path)


def test_launchers_put_the_adapter_first_with_native_line_endings(tmp_path):
    build.write_launchers(tmp_path)
    bat = (tmp_path / "mz-policy.bat").read_bytes()
    sh = (tmp_path / "mz-policy.sh").read_bytes()
    assert bat.count(b"\r\n") == bat.count(b"\n") and b"\r\r" not in bat
    assert b"\r" not in sh
    assert b'-cp "mz/*;lib/*;plugins/*" org.mage.magezero.PolicyGames' in bat
    assert b'-cp "mz/*:lib/*:plugins/*" org.mage.magezero.PolicyGames' in sh


def test_release_version_comes_from_the_core_jar(tmp_path):
    archive = tmp_path / "mage-full.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("xmage/mage-server/lib/mage-common-1.4.99.jar", b"")
        zf.writestr("xmage/mage-server/lib/mage-1.4.99.jar", b"")
    assert build.release_version(archive) == "1.4.99"
