@echo off
echo Installing required dependencies...
python -m pip install -r requirements.txt

echo.
echo Running the fully unified architecture pipeline...
python run_pipeline.py

echo.
echo Done! Check the 'abb_output_full' folder for results.
pause
