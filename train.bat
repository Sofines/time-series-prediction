@echo off
set CONFIG_FILE=config.json

:: Train the model
echo Training the model...
python training/train_model.py %CONFIG_FILE%
echo Training complete.

:: Evaluate the model
echo Evaluating the model...
python evaluation/evaluate_model.py %CONFIG_FILE%
echo Evaluation complete.

pause
