# Verifying it's working

- Call `cb_indexer_get_collection_info` to confirm the collection has a nonzero point count.
- Ask a conceptual question about the codebase and check the Claude Code transcript (`~/.claude/projects/<project-hash>/<session-id>.jsonl`) for whether it called `cb_indexer_find_repo` before `Grep`.
- Compare `/cost` on a conceptual question before and after indexing.
