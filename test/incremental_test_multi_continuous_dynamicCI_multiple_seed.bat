REM @echo off
setlocal enabledelayedexpansion

rem 设置循环次数
set "count=10"
copy D:\Ethan\Projects\DVFL\test\incremental_test_multi_continuous_dynamicCI_multiple_seed.py D:\Ethan\Projects\DVFL\incremental_test_multi_continuous_dynamicCI_multiple_seed.py
for /l %%x in (1, 1, %count%) do (
    rem python D:\Ethan\Projects\DVFL\test\mypython.py %%x
    D:\Ethan\Projects\DVFL\venv\Scripts\python.exe D:\Ethan\Projects\DVFL\incremental_test_multi_continuous_dynamicCI_multiple_seed.py %%x
)
del D:\Ethan\Projects\DVFL\incremental_test_multi_continuous_dynamicCI_multiple_seed.py