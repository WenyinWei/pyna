#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_codegraph.sh [--install-agent] [--force-index] [--status]

Initialize or refresh the local CodeGraph index for this checkout.

Options:
  --install-agent  Also run `codegraph install --target auto --location local --yes`.
                   This writes local/user agent MCP config, so it is opt-in.
  --force-index    Rebuild the project index even when .codegraph already exists.
  --status         Print CodeGraph status after setup.
  -h, --help       Show this help.

Environment:
  PYNA_CODEGRAPH_INSTALL_AGENT=1  Same as --install-agent.
EOF
}

install_agent="${PYNA_CODEGRAPH_INSTALL_AGENT:-0}"
force_index=0
show_status=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-agent)
      install_agent=1
      ;;
    --force-index)
      force_index=1
      ;;
    --status)
      show_status=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! command -v codegraph >/dev/null 2>&1; then
  echo "CodeGraph is not installed or not on PATH; skipping local index setup." >&2
  echo "Install CodeGraph first, then rerun: scripts/setup_codegraph.sh" >&2
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -d .codegraph ]]; then
  echo "Initializing CodeGraph index in $repo_root"
  codegraph init -i .
elif [[ "$force_index" == "1" ]]; then
  echo "Rebuilding CodeGraph index in $repo_root"
  codegraph index .
else
  echo "Refreshing existing CodeGraph index in $repo_root"
  codegraph sync . || codegraph index .
fi

if [[ "$install_agent" == "1" ]]; then
  echo "Installing CodeGraph MCP integration for local agent configuration"
  codegraph install --target auto --location local --yes
fi

if [[ "$show_status" == "1" ]]; then
  codegraph status .
fi
