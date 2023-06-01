set windows-shell := ["pwsh.exe", "-NoLogo", "-Command"]

run FILE="main":
    cd {{justfile_directory()}}/src && python {{FILE}}.py

ruff:
    cd {{justfile_directory()}}/src && ruff check *.py

black:
    cd {{justfile_directory()}}/src && black *.py --diff --color

mypy:
    cd {{justfile_directory()}}/src && mypy