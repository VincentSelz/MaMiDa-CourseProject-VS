from pathlib import Path


SRC = Path(__file__).parent.resolve()
ROOT = Path(__file__).parent.parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()
