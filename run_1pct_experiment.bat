@echo off
setlocal
set PYTHONPATH=src

echo ================================
echo SignMusketeers 50%% experiment
echo Windows runner
echo ================================

echo.
echo [1/6] Check data paths...
py scripts\00_check_data.py --config configs\default.yaml
if errorlevel 1 goto error

echo.
echo [2/6] Prepare 50%% manifest...
py scripts\01_prepare_manifest.py --config configs\default.yaml
if errorlevel 1 goto error

echo.
echo [3/6] Extract SignMusketeers DINOv2 features...
py scripts\02_extract_signmusketeers_dinov2_features.py --config configs\default.yaml
if errorlevel 1 goto error

echo.
echo [4/6] Train paper-like baseline model...
py scripts\03_train_t5.py --config configs\default.yaml --run_name paper_like_pretrained_dinov2 --mode paper
if errorlevel 1 goto error

echo.
echo [5/6] Train proposed CA-CSA model...
py scripts\03_train_t5.py --config configs\default.yaml --run_name ca_csa_full --mode ca_csa_full
if errorlevel 1 goto error

echo.
echo [6/6] Evaluate train/val/test and collect results...
py scripts\05_collect_results.py --config configs\default.yaml --runs paper_like_pretrained_dinov2 ca_csa_full
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