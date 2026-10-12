#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
exec java -Xmx8G -Dh2.bindAddress=127.0.0.1 -cp 'lib/*' org.mage.magezero.PolicyGames "$1"
