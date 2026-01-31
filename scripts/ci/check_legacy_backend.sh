#!/usr/bin/env bash
set -euo pipefail

token="side"
token="${token}car"
legacy_path="${token}"

if [ -e "${legacy_path}" ]; then
  echo "ERROR: legacy backend path exists: ${legacy_path}"
  exit 1
fi

pattern="(^|[^[:alnum:]_])${token}([^[:alnum:]_]|$)"
matches=0
while IFS= read -r -d '' file; do
  if [ -f "${file}" ]; then
    if grep -nE "${pattern}" "${file}"; then
      matches=1
    fi
  fi
done < <(git ls-files -z)

if [ "${matches}" -ne 0 ]; then
  echo "ERROR: legacy backend references found for token: ${token}"
  exit 1
fi

echo "No legacy backend references found."
