# Claude Code Remote Control as a systemd service

How to keep `claude remote-control` running permanently on the `openpilot-distill` server (Debian 13, user `root`), so the repositories can be driven from claude.ai/code or the Claude mobile app.

Two instances run from one systemd template unit, one per repository:

| Instance | Working directory | Session name on claude.ai/code |
| --- | --- | --- |
| `claude-rc@openpilot.distill` | `/root/openpilot.distill` | `openpilot.distill` |
| `claude-rc@sunnypilot.speedlimits` | `/root/sunnypilot.speedlimits` | `sunnypilot.speedlimits` |

The official Claude Code docs do not include a service installer, so this unit is our own wrapper around `claude remote-control`.

## Requirements

- Claude Code installed for `root` at `/root/.local/bin/claude`, recent enough to have `claude remote-control` (check with `claude remote-control --help`).
- Logged in with a claude.ai subscription account (`claude auth login`, or `/login` inside `claude`). API keys and `claude setup-token` tokens do not work with Remote Control.
- `/root/.claude/.credentials.json` must exist.

## 1. First run by hand (once per directory)

On its first run in a directory, `claude remote-control` asks whether to trust the folder and whether to enable Remote Control. systemd has no terminal to answer those questions, so run it by hand once in each directory and answer `y`. Stop it with Ctrl+C afterwards.

```bash
ssh openpilot-distill
cd /root/openpilot.distill && claude remote-control
cd /root/sunnypilot.speedlimits && claude remote-control
```

## 2. Install the template unit

```bash
cat > /etc/systemd/system/claude-rc@.service <<'EOF'
[Unit]
Description=Claude Code Remote Control (%i)
After=network-online.target
Wants=network-online.target

[Service]
User=root
WorkingDirectory=/root/%i
Environment=HOME=/root
Environment=PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/root/.local/bin/claude remote-control --name %i
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now claude-rc@openpilot.distill claude-rc@sunnypilot.speedlimits
```

`%i` is the instance name, which is the same as the directory name under `/root`. To add another repository `/root/<dir>`, do the first run by hand there (step 1), then run `systemctl enable --now claude-rc@<dir>`.

## 3. Check

```bash
systemctl status 'claude-rc@*'
journalctl -u claude-rc@openpilot.distill -f
journalctl -u claude-rc@sunnypilot.speedlimits -f
```

The two sessions should appear on claude.ai/code under their instance names.

## Operations

```bash
systemctl restart claude-rc@openpilot.distill      # after updating Claude Code
systemctl stop claude-rc@sunnypilot.speedlimits    # stop one instance
systemctl disable --now claude-rc@<dir>            # remove one instance
```

## Troubleshooting

- **`Workspace not trusted`, or the service waits at "Enable Remote Control?"**: the first run by hand was skipped or did not finish. Stop the instance, repeat step 1 for that directory, then start it again.
- **Login or subscription errors**: log in again as `root` with `claude auth login`, then restart the instances.
- **Remote Control disabled**: make sure `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`, `DISABLE_GROWTHBOOK` and a non-Anthropic `ANTHROPIC_BASE_URL` are not set, and that `disableRemoteControl` is not set in `/root/.claude/settings.json`.
- **Old unit from an earlier setup**: if a single-instance `claude-rc.service` exists, remove it:

  ```bash
  systemctl disable --now claude-rc
  rm /etc/systemd/system/claude-rc.service
  systemctl daemon-reload
  ```

## References

- https://code.claude.com/docs/en/remote-control
