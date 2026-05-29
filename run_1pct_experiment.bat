@echo off
setlocal
set PYTHONPATH=src

echo ================================
echo SignMusketeers 1%% experiment
echo Windows runner
echo ================================

echo.
echo [1/7] Check data paths...
py scripts\00_check_data.py --config configs\default.yaml
if errorlevel 1 goto error

echo.
echo [2/7] Prepare 1%% manifest...
py scripts\01_prepare_manifest.py --config configs\default.yaml
if errorlevel 1 goto error

echo.
echo [3/7] Extract SignMusketeers DINOv2 features...
py scripts\02_extract_signmusketeers_dinov2_features.py --config configs\default.yaml
if errorlevel 1 goto error

echo.
echo [4/7] Train paper-like model...
py scripts\03_train_t5.py --config configs\default.yaml --run_name paper_like_pretrained_dinov2 --mode paper
if errorlevel 1 goto error

echo.
echo [5/7] Evaluate paper-like model...
py scripts\04_evaluate_t5.py --config configs\default.yaml --checkpoint checkpoints\paper_like_pretrained_dinov2\best.pt --split test --out_csv outputs\pred_paper_like_pretrained_dinov2.csv
if errorlevel 1 goto error

echo.
echo [6/7] Train proposed confidence-aware model...
py scripts\03_train_t5.py --config configs\default.yaml --run_name proposed_confidence_aware --mode confidence
if errorlevel 1 goto error

echo.
echo [7/7] Evaluate proposed confidence-aware model...
py scripts\04_evaluate_t5.py --config configs\default.yaml --checkpoint checkpoints\proposed_confidence_aware\best.pt --split test --out_csv outputs\pred_proposed_confidence_aware.csv
if errorlevel 1 goto error

echo.
echo ================================
echo DONE!
echo Results are in outputs\
echo ================================
pause
exit /b 0

:error
echo.
echo ================================
echo ERROR occurred. Please check the message above.
echo ================================
pause
exit /b 1