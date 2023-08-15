from cx_Freeze import setup, Executable

setup(
    name = "IITD-AIIMS Anti Celiac Detection",
    version="1.0",
    author="Harshvardhan Srivastava",
    description="AI App to detect tissue, segment them, count IELs and measure image annotations",
    executables=
        [
            Executable("test_app.py")
        ]
)
