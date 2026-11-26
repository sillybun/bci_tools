# Data Analysis GUI

- Follow the repository workflow in [README.md](README.md#多台电脑同步) when committing, publishing, or syncing across computers. Start by checking the current branch, upstream, and working tree.
- On every computer, accumulate small changes into a reviewable batch. Summarize the scope and validation, and obtain explicit user approval for that batch before running `git commit` or `git push`. A single approval can cover both operations when it explicitly includes both.
- Stage only the files for the current task. Preserve pre-existing changes. Keep raw recordings, generated outputs, caches, and machine-local settings out of commits.
- Update the README in the same change when setup, defaults, external inputs, or reproducibility steps change. Identify external mapping inputs by filename and SHA-256.
- In cross-machine handoffs, give the remote, branch, verified commit SHA, required external inputs, and validation command or dataset selection.
