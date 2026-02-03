#!/usr/bin/env bash
# verify_cli.sh - Smoke tests for all librarian CLI subcommands
#
# This script verifies each subcommand runs without panics and produces
# expected help output or behavior. It tests stateless commands only,
# avoiding mutations unless explicitly isolated.
#
# Requirements:
# - Qdrant must be running (configurable via QDRANT_URL/QDRANT_API_KEY)
# - No Xinference required for --help smoke tests
#
# Usage:
#   ./scripts/ci/verify_cli.sh           # Build release and test
#   ./scripts/ci/verify_cli.sh --skip-build  # Use existing binary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="${PROJECT_ROOT}/target/release/librarian"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0

pass() { echo -e "${GREEN}✓${NC} $1"; PASSED=$((PASSED + 1)); }
fail() { echo -e "${RED}✗${NC} $1"; FAILED=$((FAILED + 1)); }
skip() { echo -e "${YELLOW}○${NC} $1 (skipped: $2)"; SKIPPED=$((SKIPPED + 1)); }

# Build release binary unless --skip-build is passed
build_binary() {
    if [[ "${1:-}" == "--skip-build" ]]; then
        if [[ ! -x "$BINARY" ]]; then
            echo "ERROR: Binary not found at $BINARY and --skip-build specified"
            exit 1
        fi
        echo "Using existing binary: $BINARY"
        return
    fi
    echo "Building release binary..."
    cargo build --release --manifest-path "$PROJECT_ROOT/Cargo.toml"
}

# Test that a command exits 0 and its output contains expected text
test_help() {
    local cmd="$1"
    local desc="${2:-$cmd --help}"
    if $BINARY $cmd --help 2>&1 | grep -q "Usage:"; then
        pass "$desc"
    else
        fail "$desc"
    fi
}

# Test that a command exits 0 (no panic)
test_runs() {
    local cmd="$1"
    local desc="$2"
    local timeout_sec="${3:-10}"
    if timeout "$timeout_sec" $BINARY $cmd >/dev/null 2>&1; then
        pass "$desc"
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            fail "$desc (timeout)"
        else
            # Non-zero exit may be expected (e.g. missing config) - check for panic
            if timeout "$timeout_sec" $BINARY $cmd 2>&1 | grep -qi "panic"; then
                fail "$desc (panic detected)"
            else
                pass "$desc (non-zero exit but no panic)"
            fi
        fi
    fi
}

# Test that command output contains expected text
test_output_contains() {
    local cmd="$1"
    local expected="$2"
    local desc="$3"
    local output
    # shellcheck disable=SC2086
    output=$($BINARY $cmd 2>&1) || true
    if echo "$output" | grep -q "$expected"; then
        pass "$desc"
    else
        fail "$desc (expected '$expected' not found)"
    fi
}

main() {
    echo "====================================="
    echo "librarian CLI Verification Suite"
    echo "====================================="
    echo ""

    build_binary "${1:-}"

    echo ""
    echo "--- Top-level help ---"
    test_help "" "librarian --help"
    test_output_contains "--version" "librarian" "librarian --version"

    echo ""
    echo "--- Subcommand help (stateless) ---"
    test_help "init"
    test_help "ingest"
    test_help "ingest dir"
    test_help "ingest url"
    test_help "ingest sitemap"
    test_help "query"
    test_help "status"
    test_help "sources"
    test_help "prune"
    test_help "reindex"
    test_help "update"
    test_help "remove"
    test_help "rename"
    test_help "mcp"
    test_help "completions"
    test_help "db"
    test_help "db init"
    test_help "db status"
    test_help "db check"
    test_help "db reset"
    test_help "config"
    test_help "config print"
    test_help "xinference"
    test_help "xinference sync-models"

    echo ""
    echo "--- Completions (stateless) ---"
    test_output_contains "completions bash" "_librarian" "completions bash generates output"
    test_output_contains "completions zsh" "_librarian" "completions zsh generates output"
    test_output_contains "completions fish" "librarian" "completions fish generates output"

    echo ""
    echo "--- Config print (stateless) ---"
    test_output_contains "config print" "qdrant" "config print shows qdrant section"

    echo ""
    echo "====================================="
    echo "Summary"
    echo "====================================="
    echo -e "Passed:  ${GREEN}${PASSED}${NC}"
    echo -e "Failed:  ${RED}${FAILED}${NC}"
    echo -e "Skipped: ${YELLOW}${SKIPPED}${NC}"
    echo ""

    if [[ $FAILED -gt 0 ]]; then
        echo -e "${RED}FAILED${NC}: $FAILED test(s) failed"
        exit 1
    else
        echo -e "${GREEN}PASSED${NC}: All tests passed"
        exit 0
    fi
}

main "$@"
