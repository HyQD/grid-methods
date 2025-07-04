#!/bin/bash
set -x
set -e
################################################################################
# File:    publishDocs.sh
# Purpose: Script that builds our documentation using sphinx and updates GitHub
#          Pages. This script is executed by:
#            .github/workflows/docs_pages.yml
#
# Authors: Michael Altfield <michael@michaelaltfield.net>
# Created: 2020-07-17
# Updated: 2022-01-07
# Version: 0.1.1
################################################################################

#####################
# DECLARE VARIABLES #
#####################

export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)

#######################
# Update GitHub Pages #
#######################

git config --global user.name "${GITHUB_ACTOR}"
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"

docroot=`mktemp -d`
rsync -av "docs/_build/" "${docroot}/"

pushd "${docroot}"

# don't bother maintaining history; just generate fresh
git init
git remote add deploy "https://token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git checkout -b gh-pages

# add .nojekyll to the root so that github won't 404 on content added to dirs
# that start with an underscore (_), such as our "_content" dir..
touch .nojekyll

# Add README
cat > README.md <<EOF

### GitHub Pages Cache

Nothing to see here. The contents of this branch are essentially a cache that's not intended to be viewed on github.com.

For more information on how this documentation is built using Sphinx, Read the Docs, and GitHub Actions/Pages, see:

 * https://tech.michaelaltfield.net/2020/07/18/sphinx-rtd-github-pages-1
EOF

# Add index.html
cat > index.html <<EOF
<!DOCTYPE html>
<html>
  <head>
    <title>Redirecting to main branch</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=./main/index.html">
    <link rel="canonical" href="https://hyqd.github.io/grid-lib/main/index.html">
  </head>
</html>
EOF

# copy the resulting html pages built from sphinx above to our new git repo
git add .

# commit all the new files
msg="Updating Docs for commit ${GITHUB_SHA} made on `date -d"@${SOURCE_DATE_EPOCH}" --iso-8601=seconds` from ${GITHUB_REF} by ${GITHUB_ACTOR}"
git commit -am "${msg}"

# overwrite the contents of the gh-pages branch on our github.com repo
git push deploy gh-pages --force

popd # return to main repo sandbox root

# exit cleanly
exit 0
