import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats

from scipy import interpolate

plt.style.use('ggplot')

def barplot(data:pd.DataFrame, feature):
    return data[feature].value_counts().plot.bar()
