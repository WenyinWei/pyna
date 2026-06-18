<!-- CODEGRAPH_START -->
## CodeGraph

`.codegraph/` is a local, regenerable code-intelligence index and is ignored by
git. Do not commit it.

When working in this repository:

- If `.codegraph/` exists, use CodeGraph before grep/find or broad file reads
  when locating or understanding code.
- If `.codegraph/` is missing and `codegraph` is available on `PATH`, run:

  ```bash
  scripts/setup_codegraph.sh
  ```

- To connect CodeGraph to local agent MCP configuration, run this explicitly:

  ```bash
  scripts/setup_codegraph.sh --install-agent
  ```

  This delegates target detection to `codegraph install --target auto` and may
  write local/user agent config, so it is intentionally opt-in.

Useful commands:

```bash
codegraph explore "<question or symbols>"
codegraph node <symbol-or-file>
codegraph sync .
```
<!-- CODEGRAPH_END -->
