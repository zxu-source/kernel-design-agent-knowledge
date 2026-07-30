# Hosted-code collection

Use official provider APIs, not rendered repository pages. The first reusable
adapter is GitCode merged-PR collection:

```bash
python3 scripts/collect_gitcode_prs.py \
  --repo AI4Science/dgl-ascend \
  --limit 20 \
  --output data/crawl-runs/gitcode/AI4Science__dgl-ascend.json
```

Set `GITCODE_TOKEN` to a read-only personal access token for endpoints or
repositories that require authorization. Do not put a token in a command line,
config file, source page, or artifact.

The JSON output is *staged evidence*, not automatically trusted knowledge. A
subsequent importer must classify a PR, pin its merged SHA, fetch selected
source files by SHA, create a source-change page, write an artifact manifest,
and run the normal validator. This separation prevents a broad discovery run
from silently becoming authoritative KernelWiki knowledge.
