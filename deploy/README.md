# Deploying an AVIR verification network

Turnkey persistence for the coordinator and verifier nodes. macOS uses
`launchd` (`setup_launchd.sh`); Linux operators can adapt the same commands into
a systemd unit.

## Coordinator (host the job board)

```bash
pip install -e ".[network,crypto]"          # in your venv
COORD_HOST=0.0.0.0 COORD_PORT=8770 SAN="my.host 100.x.y.z" \
  bash deploy/setup_launchd.sh coordinator
```

This generates a self-signed TLS cert (SAN = localhost + your SAN hosts), a
random submission token, writes `~/Library/LaunchAgents/com.avir.coordinator.plist`,
and loads it (RunAtLoad + KeepAlive, restarts on reboot/crash). It prints the CA
bundle path to hand to verifiers and the token path.

Exposure is your choice via `COORD_HOST`:
- `127.0.0.1` (default) -- local only.
- a Tailscale/LAN IP -- reachable by your devices / cluster.
- `0.0.0.0` behind a port-forward or reverse proxy -- internet-reachable
  (use a publicly-issued cert for that; self-signed is fine for LAN/Tailscale).

Uninstall: `launchctl unload ~/Library/LaunchAgents/com.avir.coordinator.plist && rm` it.

## Verifier (contribute your model)

```bash
VERIFIER_COORD=https://my.host:8770 VERIFIER_PROVIDER=claude-cli VERIFIER_ORG=alice \
AVIR_CA_BUNDLE=~/.avir/coordinator.crt \
  bash deploy/setup_launchd.sh verifier

# or an OpenAI-compatible endpoint you run:
VERIFIER_COORD=https://my.host:8770 VERIFIER_PROVIDER=http \
VERIFIER_ENDPOINT=http://localhost:11434 VERIFIER_MODEL=llama3 VERIFIER_ORG=alice \
AVIR_CA_BUNDLE=~/.avir/coordinator.crt \
  bash deploy/setup_launchd.sh verifier
```

The verifier polls forever (KeepAlive), judging jobs with your provider and
submitting Ed25519-signed verdicts. Your provider credentials never leave your
machine.

## Security checklist

- TLS on (cert + key); clients trust it via `AVIR_CA_BUNDLE` (or a public CA).
- `AVIR_COORDINATOR_TOKEN` set (the script does this) to gate job submission.
- Identity/token/key files are written `0600`.
- Prefer Tailscale or a reverse proxy over raw `0.0.0.0` on the open internet.
