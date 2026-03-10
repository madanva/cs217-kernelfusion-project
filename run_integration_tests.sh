#!/bin/bash
# Run attention integration tests for all 3 variants through GBModule
# Must be run from the cs217-project directory

set -e

export REPO_TOP="$(pwd)"
export LD_LIBRARY_PATH=/cad/mentor/2024.2_1/Mgc_home/shared/lib:$LD_LIBRARY_PATH

TESTDIR="src/Top/GBPartition/GBModule"
OUTDIR="reports/integration"
mkdir -p "$OUTDIR"

echo "============================================"
echo " Attention Integration Tests (GBModule)"
echo " $(date)"
echo "============================================"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

for VARIANT in UNFUSED PARTIAL_FUSED FULLY_FUSED; do
  echo "--------------------------------------------"
  echo " Building: $VARIANT"
  echo "--------------------------------------------"

  if [ "$VARIANT" = "UNFUSED" ]; then
    FLAGS="-DATTN_UNFUSED"
  elif [ "$VARIANT" = "PARTIAL_FUSED" ]; then
    FLAGS="-DATTN_PARTIAL_FUSED"
  else
    FLAGS=""
  fi

  cd "$TESTDIR"
  rm -f sim_test_attn

  make sim_test_attn USER_FLAGS="$FLAGS" 2>&1
  BUILD_RC=$?

  if [ $BUILD_RC -ne 0 ]; then
    echo "BUILD FAILED for $VARIANT"
    cat build.log
    FAIL_COUNT=$((FAIL_COUNT + 1))
    cd "$REPO_TOP"
    continue
  fi

  echo "Build OK, running..."
  LOGFILE="$REPO_TOP/$OUTDIR/integration_${VARIANT}.log"
  ./sim_test_attn > "$LOGFILE" 2>&1
  RUN_RC=$?

  if [ $RUN_RC -eq 0 ]; then
    echo "PASS: $VARIANT"
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    echo "FAIL: $VARIANT (exit code $RUN_RC)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi

  # Show summary from log
  echo ""
  tail -10 "$LOGFILE"
  echo ""

  cd "$REPO_TOP"
done

echo "============================================"
echo " RESULTS: $PASS_COUNT passed, $FAIL_COUNT failed"
echo " Logs in: $OUTDIR/"
echo " $(date)"
echo "============================================"

exit $FAIL_COUNT
