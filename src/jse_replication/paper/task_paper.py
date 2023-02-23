import shutil

import pytask

from jse_replication.config import BLD
from jse_replication.config import ROOT
from jse_replication.config import SRC


documents = ["term_paper"]


#@pytask.mark.latex(
#    [
#        "--pdf",
#        "--interaction=nonstopmode",
#        "--synctex=1",
#        "--cd",
#        "--quiet",
#        "--shell-escape",
#    ]
#)
#@pytask.mark.parametrize(
#    "depends_on, produces",
#    [
#        (SRC / "paper" / f"{document}.tex", BLD / "paper" / f"{document}.pdf")
#        for document in documents
#    ],
#)
@pytask.mark.latex(script="term_paper.tex", document="term_paper.pdf")
def task_compile_documents():
    pass


#    "depends_on, produces",
#    [
#        (BLD / "paper" / f"{document}.pdf", ROOT / f"{document}.pdf")
#        for document in documents
#    ],
#)
#def task_copy_to_root(depends_on, produces):
#    shutil.copy(depends_on, produces)
