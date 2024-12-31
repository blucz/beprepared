#!/usr/bin/env bash

set -e -o pipefail

echo "Entering beprepared/web directory"
pushd beprepared/web

# Track if we moved .env
moved_env=false

# Setup cleanup trap
cleanup() {
    if $moved_env && [ -f .env.bak_for_buildcommitweb ]; then
        echo "Restoring .env file from backup"
        mv .env.bak_for_buildcommitweb .env
    fi
}
trap cleanup EXIT

# Backup .env if it exists
if [ -f .env ]; then
    echo "Backing up .env file to .env.bak_for_buildcommitweb"
    mv .env .env.bak_for_buildcommitweb
    moved_env=true
fi

echo "Building web assets..."
npm run build

popd

echo "Adding static files to git..."
git add beprepared/web/static

echo "Committing web assets..."
git commit -m 'Build web assets' beprepared/web/static

echo "Web assets build and commit completed successfully"

