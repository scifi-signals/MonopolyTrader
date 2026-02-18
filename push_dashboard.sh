#!/bin/bash
cd /root/monopoly-trader
git pull --rebase origin main 2>/dev/null || git reset --hard origin/main
if [[ -n $(git diff --name-only dashboard/data.json 2>/dev/null) ]]; then
    git add dashboard/data.json
    git commit -m "Dashboard update $(date +'%Y-%m-%d %H:%M')"
    git push origin main
fi
