"""Read in the SCE data, rename columns and harmonize values."""

import pytask
import pandas as pd
import numpy as np

from jse_replication.config import BLD,SRC

str_dep1 = "data/FRBNY-SCE-Public-Microdata-Complete-13-16.xlsx"
str_dep2 = "data/FRBNY-SCE-Public-Microdata-Complete-17-19.xlsx"

@pytask.mark.depends_on([
    SRC / "data" / "FRBNY-SCE-Public-Microdata-Complete-13-16.xlsx",
    SRC / "data" / "FRBNY-SCE-Public-Microdata-Complete-17-19.xlsx"
    ])
@pytask.mark.produces(BLD / "data" / "merged_SCE_data.csv")
def task_merge_data(depends_on,produces):
    """Read data, clean data adn export the data."""
    df1 = pd.read_excel(depends_on[0],header=1)
    df2 = pd.read_excel(depends_on[1],header=1)
    for df in [df1,df2]:
        # Save as proper datetime format
        df['date'] = df['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m'))
        # Set index
        df.set_index(['userid','date'],inplace=True)
    # merge  data
    df = pd.concat([df1, df2])
    # export data
    df.to_csv(produces)


