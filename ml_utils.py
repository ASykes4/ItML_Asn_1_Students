import pandas as pd
import numpy as np
import math
import sklearn.datasets
import ipywidgets as widgets

##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8,8)

class edaDF:
    """
    A class used to perform common EDA tasks

    ...

    Attributes
    ----------
    data : dataframe
        a dataframe on which the EDA will be performed
    target : str
        the name of the target column
    cat : list
        a list of the names of the categorical columns
    num : list
        a list of the names of the numerical columns
    outliers : pandas DataFrame
        a DataFrame containing rows that correspond to each column with the low and high outliers naively generated 

    Methods
    -------
    setCat(catList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        catlist : list
            The list of column names that are categorical

    setNum(numList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        numlist : list
            The list of column names that are numerical

    countPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned.
    
    histPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned. 

    autoFeatures()
        fills the cat and num variables with proper feature lists by dynamically obtaining rows that are 
        category or object for cat, and int64 and float64 for num.

    outlierCounter()
        generates the DataFrame of outliers. It calculates the outliers by taking the power of 10 lower than the 
        25th quartile as its low cutoff and the power of 10 higher than the 75th quartile as its high cutoff.

    fullEDA()
        Displays the full EDA process. 
    """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cat = []
        self.num = []
        self.autoFeatures()
        self.outliers = self.outlierCounter()

    def info(self):
        return self.data.info()

    def giveTarget(self):
        return self.target
        
    def setCat(self, catList):
        self.cat = catList
    
    def setNum(self, numList):
        self.num = numList

    def debug(self):
        print('Debuggerating')

    def autoFeatures(self):
        self.cat = list(self.data.select_dtypes(['object', 'category']).columns)
        self.num = list(self.data.select_dtypes(['int64', 'float64']).columns)
        if self.cat.count(self.target) > 0:
            self.cat.remove(self.target)
        if self.num.count(self.target) > 0:
            self.num.remove(self.target)

    def countPlots(self, splitTarg=False, show=True):
        n = len(self.cat)
        cols = 2
        if n != 0:
            figure, ax = plt.subplots(math.ceil(n/cols), cols, figsize=(20,20))
            r = 0
            c = 0
            for col in self.cat:
                if splitTarg == False:
                    sns.countplot(data=self.data, x=col, ax=ax[r][c])
                if splitTarg == True:
                    sns.countplot(data=self.data, x=col, hue=self.target, ax=ax[r][c])
                c += 1
                if c == cols:
                    r += 1
                    c = 0
            if show == True:
                figure.tight_layout()
                figure.show()
            return figure
        else:
            print("No categorical features in the data.")

    def histPlots(self, kde=True, splitTarg=False, show=True):
        n = len(self.num)
        cols = 2
        if n != 0:
            figure, ax = plt.subplots(math.ceil(n/cols), cols, figsize=(20,20))
            r = 0
            c = 0
            for col in self.num:
                #print("r:",r,"c:",c)
                if splitTarg == False:
                    sns.histplot(data=self.data, x=col, kde=kde, ax=ax[r][c])
                if splitTarg == True:
                    sns.histplot(data=self.data, x=col, hue=self.target, kde=kde, ax=ax[r][c])
                c += 1
                if c == cols:
                    r += 1
                    c = 0
            if show == True:
                figure.tight_layout()
                figure.show()
            return figure
        else:
            print("No numerical features in the data.")

    def outlierCounter(self):
        outliers = pd.DataFrame(columns=["Feature", "Low Outliers", "High Outliers"])
        for col in self.num:
            lo = 0
            ho = 0
            loweval = self.data[col].quantile(0.25)*0.1
            higheval = self.data[col].quantile(0.75)*10
            if higheval == 0:
                higheval = 1
            for i in self.data[col]:
                if i < loweval and i != 0:
                    lo+=1
                if i >= higheval:
                    ho+=1
            outliers = pd.concat([outliers,pd.Series({"Feature":col,"Low Outliers":lo,"High Outliers":ho}).to_frame().T], ignore_index=True)
        return outliers

    def fullEDA(self):
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()
        out5 = widgets.Output()
        out6 = widgets.Output()
        out7 = widgets.Output()
        out8 = widgets.Output()
        out9 = widgets.Output()

        tab = widgets.Tab(children=[out1,out2,out3,out4,out5,out6,out7,out8,out9])
        tab.set_title(0,"Info")
        tab.set_title(1,"Categorical")
        tab.set_title(2,"Numerical")
        tab.set_title(3,"Outlier Counts")
        tab.set_title(4,"Descriptive Stats")
        tab.set_title(5,"Value Counts")
        tab.set_title(6,"Correlations")
        tab.set_title(7,"Missing Values")
        tab.set_title(8,"Pairplot")
        display(tab)

        with out1:
            display(pd.DataFrame(self.info()))

        with out2:
            fig2 = self.countPlots(splitTarg=True, show=False)
            plt.show(fig2)
        
        with out3:
            fig3 = self.histPlots(kde=True, show=False)
            plt.show(fig3)

        with out9:
            fig4 = sns.pairplot(self.data)
            plt.show(fig4)
        
        with out5:
            display(pd.DataFrame(self.data.describe()))

        with out6:
            for col in self.cat:
                display(pd.DataFrame(self.data[col].value_counts()))
                print('\n')

        with out7:
            corr = self.data.corr()
            heat = sns.heatmap(corr, annot=True)
            plt.show(heat)

        with out8:
            display(pd.DataFrame(self.data.isnull().sum()))

        with out4:
            print("These outliers are calculated by taking powers of 10 of the 25th and 75th quartiles, and are a guideline only")
            display(self.outliers)
