#!/usr/bin/env bash
# Stand up a persistent AVIR coordinator (and optionally a verifier) as macOS
# launchd services, with TLS and a submission token. Reversible (see UNINSTALL).
#
# Usage:
#   COORD_HOST=0.0.0.0 COORD_PORT=8770 \
#   SAN="my.host.name 100.x.y.z" \
#   bash deploy/setup_launchd.sh coordinator
#
#   VERIFIER_COORD=https://my.host:8770 VERIFIER_PROVIDER=claude-cli VERIFIER_ORG=alice \
#   AVIR_CA_BUNDLE=~/.avir/coordinator.crt \
#   bash deploy/setup_launchd.sh verifier
#
# Requires the venv that has avir-protocol[network,crypto] installed; set
# PYTHON to its python (default: ./.venv/bin/python).
set -euo pipefail

ROLE="${1:-coordinator}"
PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
AVIR_DIR="${AVIR_DIR:-$HOME/.avir}"
LA_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$AVIR_DIR" "$LA_DIR"

if [ ! -x "$PYTHON" ]; then echo "PYTHON not executable: $PYTHON"; exit 1; fi

if [ "$ROLE" = "coordinator" ]; then
  COORD_HOST="${COORD_HOST:-127.0.0.1}"
  COORD_PORT="${COORD_PORT:-8770}"
  CERT="$AVIR_DIR/coordinator.crt"; KEY="$AVIR_DIR/coordinator.key"
  DB="$AVIR_DIR/coordinator.db"
  TOKEN_FILE="$AVIR_DIR/coordinator.token"

  # TLS cert (covers localhost + any SAN hosts provided)
  HOSTS=(localhost 127.0.0.1); for h in ${SAN:-}; do HOSTS+=("$h"); done
  HOST_ARGS=(); for h in "${HOSTS[@]}"; do HOST_ARGS+=(--host "$h"); done
  "$PYTHON" -m avir.network.tls "${HOST_ARGS[@]}" --cert "$CERT" --key "$KEY"

  # submission token
  if [ ! -f "$TOKEN_FILE" ]; then "$PYTHON" -c "import secrets;print(secrets.token_hex(24))" > "$TOKEN_FILE"; chmod 600 "$TOKEN_FILE"; fi
  TOKEN="$(cat "$TOKEN_FILE")"

  PLIST="$LA_DIR/com.avir.coordinator.plist"
  cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.avir.coordinator</string>
  <key>ProgramArguments</key><array>
    <string>$PYTHON</string><string>-m</string><string>avir.network.coordinator</string>
    <string>--host</string><string>$COORD_HOST</string>
    <string>--port</string><string>$COORD_PORT</string>
    <string>--db</string><string>$DB</string>
    <string>--tls-cert</string><string>$CERT</string>
    <string>--tls-key</string><string>$KEY</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>AVIR_COORDINATOR_TOKEN</key><string>$TOKEN</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$AVIR_DIR/coordinator.out.log</string>
  <key>StandardErrorPath</key><string>$AVIR_DIR/coordinator.err.log</string>
</dict></plist>
PL
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load "$PLIST"
  echo "coordinator loaded: https://$COORD_HOST:$COORD_PORT  (TLS, token-gated)"
  echo "  token:      $TOKEN_FILE"
  echo "  CA bundle:  $CERT  (give to verifiers: export AVIR_CA_BUNDLE=$CERT)"
  echo "  UNINSTALL:  launchctl unload $PLIST && rm $PLIST"

elif [ "$ROLE" = "verifier" ]; then
  : "${VERIFIER_COORD:?set VERIFIER_COORD=https://host:port}"
  VERIFIER_PROVIDER="${VERIFIER_PROVIDER:-claude-cli}"
  VERIFIER_ORG="${VERIFIER_ORG:-independent}"
  EXTRA=()
  [ -n "${VERIFIER_ENDPOINT:-}" ] && EXTRA+=(--endpoint "$VERIFIER_ENDPOINT")
  [ -n "${VERIFIER_MODEL:-}" ] && EXTRA+=(--model "$VERIFIER_MODEL")
  EXTRA_XML=""; for e in "${EXTRA[@]:-}"; do [ -n "$e" ] && EXTRA_XML="$EXTRA_XML<string>$e</string>"; done

  PLIST="$LA_DIR/com.avir.verifier.plist"
  cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.avir.verifier</string>
  <key>ProgramArguments</key><array>
    <string>$PYTHON</string><string>-m</string><string>avir.network.verifier</string>
    <string>--coordinator</string><string>$VERIFIER_COORD</string>
    <string>--provider</string><string>$VERIFIER_PROVIDER</string>
    <string>--org</string><string>$VERIFIER_ORG</string>
    $EXTRA_XML
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>AVIR_CA_BUNDLE</key><string>${AVIR_CA_BUNDLE:-}</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$AVIR_DIR/verifier.out.log</string>
  <key>StandardErrorPath</key><string>$AVIR_DIR/verifier.err.log</string>
</dict></plist>
PL
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load "$PLIST"
  echo "verifier loaded -> $VERIFIER_COORD (provider=$VERIFIER_PROVIDER org=$VERIFIER_ORG)"
  echo "  UNINSTALL: launchctl unload $PLIST && rm $PLIST"
else
  echo "usage: setup_launchd.sh [coordinator|verifier]"; exit 1
fi
