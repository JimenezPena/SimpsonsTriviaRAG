#!/usr/bin/env bash
# Render build step: install dependencies and fetch the vector DB from the
# GitHub release (it is git-ignored because it is too heavy for the repo).
set -o errexit

pip install --upgrade pip
pip install -r requirements-web.txt

DB_URL="https://github.com/JimenezPena/SimpsonsTriviaRAG/releases/download/latest-db/db.tar.gz"

rm -rf batch_vector_db
curl -fSL --retry 3 -o db.tar.gz "$DB_URL"
tar -xzf db.tar.gz          # extracts to ./batch_vector_db/
rm -f db.tar.gz

test -f batch_vector_db/chroma.sqlite3 || { echo "chroma.sqlite3 not found after extraction"; exit 1; }
echo "Vector DB ready: $(du -sh batch_vector_db | cut -f1)"
