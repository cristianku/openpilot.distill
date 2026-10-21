"""distill-web: the always-on control site of openpilot.distill.

Start it from the repo root with the repo python: python -m web.server --host 0.0.0.0 --port 8766
(helpers/web.sh does it for you, deploy/install_web_service.sh installs it as a systemd service).
Standard library only: the site must keep working when the training environment is broken.
"""
