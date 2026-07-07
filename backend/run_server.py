import uvicorn
import sys


def print_runtime() -> None:
    print(f"Python executable: {sys.executable}")
    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch: not installed in this environment")
        return
    print(f"PyTorch: {torch.__version__}")


if __name__ == "__main__":
    print_runtime()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, log_level="info")
