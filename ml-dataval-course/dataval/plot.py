import pandas as pd
import seaborn as sns

def violinplot(data: pd.DataFrame, variable: str, groupby: str = "week"):
    if groupby == "week":
        return sns.violinplot(data = data, x=variable, y=data["fact_time"].dt.isocalendar().week.astype(str))
    
    if groupby == "day":
        return sns.violinplot(data = data, x=variable, y=data["fact_time"].dt.isocalendar().day.astype(str))
    
    if groupby == "hour":
        return sns.violinplot(data = data, x=variable, y=data["fact_time"].dt.hour.astype(str))
    
    if groupby == "hour-binary":
        return sns.violinplot(data = data, x=variable, y=data["fact_time"].dt.hour.between(10, 17).astype(str))
    
    raise ValueError("groupby must be week, day, or hour.")