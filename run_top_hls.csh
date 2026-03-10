#!/bin/csh -f
# Top-level HLS synthesis for all 3 attention variants
# Run inside rhel8 container after: source setup.csh
# Each variant: GBModule -> GBPartition -> Top (bottom-up)
# Output saved to reports/hls_top/<variant>/

set PROJDIR = /home/users/vmadan/cs217-project
set OUTDIR = $PROJDIR/reports/hls_top
set HLSDIR = $PROJDIR/hls/Top

cd $PROJDIR
source setup.csh

mkdir -p $OUTDIR

echo "============================================"
echo " Top-Level HLS Synthesis - All 3 Variants"
echo " `date`"
echo "============================================"
echo ""

foreach VARIANT (AttnUnfused AttnPartialFused AttnFullyFused)
  echo "============================================"
  echo " VARIANT: $VARIANT"
  echo " `date`"
  echo "============================================"

  # Set environment for TCL scripts
  setenv ATTN_VARIANT $VARIANT

  # Set COMPILER_FLAGS based on variant
  if ($VARIANT == "AttnUnfused") then
    set CFLAGS = "HLS_ALGORITHMICC ATTN_UNFUSED"
  else if ($VARIANT == "AttnPartialFused") then
    set CFLAGS = "HLS_ALGORITHMICC ATTN_PARTIAL_FUSED"
  else
    set CFLAGS = "HLS_ALGORITHMICC"
  endif

  mkdir -p $OUTDIR/$VARIANT

  # Step 1: Synthesize GBModule with this variant
  echo "--- Step 1: GBModule ---"
  cd $HLSDIR/GBPartition/GBModule
  make clean
  make hls COMPILER_FLAGS="$CFLAGS" |& tee $OUTDIR/$VARIANT/gbmodule.log
  if ($status != 0) then
    echo "FAILED: GBModule for $VARIANT"
    cd $PROJDIR
    continue
  endif
  echo "GBModule done for $VARIANT"

  # Step 2: Synthesize GBPartition
  echo "--- Step 2: GBPartition ---"
  cd $HLSDIR/GBPartition
  make clean
  make hls COMPILER_FLAGS="$CFLAGS" |& tee $OUTDIR/$VARIANT/gbpartition.log
  if ($status != 0) then
    echo "FAILED: GBPartition for $VARIANT"
    cd $PROJDIR
    continue
  endif
  echo "GBPartition done for $VARIANT"

  # Step 3: Synthesize Top
  echo "--- Step 3: Top ---"
  cd $HLSDIR
  make clean
  make hls COMPILER_FLAGS="$CFLAGS" |& tee $OUTDIR/$VARIANT/top.log
  if ($status != 0) then
    echo "FAILED: Top for $VARIANT"
    cd $PROJDIR
    continue
  endif
  echo "Top done for $VARIANT"

  # Step 4: Save the synthesized RTL output
  if (-f Catapult/Top.v1/concat_Top.v) then
    cp Catapult/Top.v1/concat_Top.v $OUTDIR/$VARIANT/concat_Top.v
    echo "Saved concat_Top.v for $VARIANT"
  else
    echo "WARNING: concat_Top.v not found for $VARIANT"
  endif

  echo "DONE: $VARIANT at `date`"
  echo ""
  cd $PROJDIR
end

echo "============================================"
echo " ALL VARIANTS COMPLETE"
echo " `date`"
echo "============================================"
