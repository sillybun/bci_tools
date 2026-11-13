# Code Relation Graph

| Source | Relation | Target | Contract |
| --- | --- | --- | --- |
| `models.AcceptanceReview` | data-contract | `.codex/commit-acceptance/<sha>.json` | Schema version, commit, timestamp, ordered item/answer pairs, notes |
| `github-collaboration-safety` | sync | `git_service.ACCEPTANCE_ITEMS` | The eleven safety-table row names stay identical |
| `git_service.acceptance_review_prompt` | data-contract | `.codex/commit-acceptance/pending.json` | Prompt embeds the eleven ordered rows, schema, review-only safety boundary, and resolved repository path |
| `MainWindow.refresh_all.progress` | navigation | `MainWindow.details` | Background repository updates may refresh metadata but only explicit repository selection may leave an open commit detail |
| `MainWindow._poll_worktree_commits` | refresh | `git_service.list_commits` / `MainWindow._request_commits` | Polls the current repository every 1.5s; worktree pseudo-commit appearance, disappearance, and selected diff stay aligned with the filesystem |
| `MainWindow._poll_selected_acceptance` | refresh | `git_service.load_acceptance` / `.codex/commit-acceptance/pending.json` | Reloads an externally changed pending review while preserving unchanged in-progress edits |
| `CloneDialog` | interface | `clone_repository` | `none` or one approved prefix; non-none requires branch leaf |
| `inspect_repository` | upstream | repository list | State precedence is disconnected, dirty, stale, synced |
| streaming refresh generation / current repository | consistency | repository list | Each repository result updates its row immediately; only the newest async refresh may publish progress, and the current selection remains stable |
| local-first refresh / bounded concurrent fetch | performance | repository list | Publish local state for every repository before at most three concurrent remote fetches; coalesce in-flight focus refreshes and apply a 30-second focus cooldown |
| repository item roles | interface | `RepoStateDelegate` | State background and foreground remain visible when selected |
| `list_commits` | upstream | commit list/detail page | Empty SHA denotes the tracked-worktree pseudo-commit; empty parent/ref fields remain valid fields during Git log parsing |
| `rev-list` reachability sets | performance | commit synchronization states | Two batched reachability queries replace per-commit `merge-base` subprocesses without changing push/pull state semantics |
| `CommitInfo` state flags | interface | `CommitRow` | Worktree, remote-only, and local-only cards expose only their applicable action |
| `CommitInfo.authored_at` | interface | `CommitRow.time_label` | Every real commit card shows its ISO timestamp as local wall time; the worktree pseudo-commit has no timestamp label |
| `MainWindow.selected_commit` | consistency | `CommitRow.surface[selected]` / `MainWindow._update_commit_selection` | The card corresponding to the open detail remains visibly selected; all other commit cards remain unselected |
| `MainWindow._load_commit_diff` generation | refresh | `changed_files` / `diff_text` / `diffProgress` | Worktree diff computation runs off the GUI thread, shows loading feedback, and stale results cannot overwrite a newer selected commit |
| commit parent SHAs | data-contract | `BranchGlyph` | Each row carries lane transitions; merge commits fan out to every parent |
| current `HEAD` | scope | contributor list | Unrelated remote/upstream refs must not contribute authors |
| repository README | upstream | repository detail | Markdown renders locally with the repository root as the relative-resource base |
| `RepoStore` | configuration | `MainWindow` | Only paths are persisted; credentials and repository contents are never copied |
| `CredentialDialog` | interface | `CredentialStore` | Username/email identity plus manual token stored in app-private plaintext file, or external token file path referenced; no OS keyring |
| `CredentialStore` | interface | `git_service._run` / `app.main` | Host-scoped Git helper override and askpass; HTTP requires explicit opt-in |
| `CredentialStore.identity_for_remote` | interface | `MainWindow._show_repo` | Map both HTTP(S) and SSH origins to the host-scoped configured username/email; render SSH as plain text `SSH:user<email>` so the fixed `git` transport user is never exposed as an account |
| repository settings panel | interface | `RepoStore` / `configure_origin` / `switch_branch` | Local path rebinding updates app state; origin and branch controls mutate the selected Git repository only after confirmation |
| `list_branches` | data-contract | repository branch selector / `branch_switch_args` | Merge local and fetched `origin/*` names; remote-only names map to `git switch --track -c` |
| unborn `HEAD` state | consistency | `inspect_repository` / `has_tracked_changes` / commit list | A repository without a first commit is dirty and includes initial untracked files in its worktree pseudo-commit |
| add-branch menu item | interface | `create_branch_args` / `create_branch` | Prompted name must be a valid, non-existing ref before `git switch -c` confirmation |
| `RepoStateDelegate` | consistency | repository state surface | Selection outline reuses the exact rounded state-background rectangle |
| `assets/app-icon-source-v3.png` | generated-by | `imagegen` | Source artwork keeps transparent margin, muted highlights, and recognizable branch motif |
| `packaging/generate_icons.py` | sync | native packages | PNG, ICNS, and ICO derive from the same source artwork |
| `packaging/*` | configuration | `pyproject.toml` | Native target builds install the same project, Material fonts, icon, and PyInstaller entry point |
