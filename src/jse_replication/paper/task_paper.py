import shutil

import pytask

from jse_replication.config import BLD
from jse_replication.config import ROOT
from jse_replication.config import SRC

documents = ['term_paper']

@pytask.mark.latex(script="term_paper.tex", document="term_paper.pdf")
def task_compile_documents( ):
    pass

@pytask.mark.parametrize("depends_on, produces",
    [
        (SRC / "paper" / f"{document}.pdf", BLD / "paper" / f"{document}.pdf")
        for document in documents
    ],
)
def task_copy_to_bld(depends_on, produces):
    shutil.copy(depends_on, produces)

@pytask.mark.parametrize("depends_on, produces",
    [
        (SRC / "paper" / f"{document}.pdf", ROOT / f"VincentSelz7707241_TermPaper_MaMiDa022023.pdf")
        for document in documents
    ],
)
def task_copy_to_root(depends_on, produces):
    shutil.copy(depends_on, produces)
