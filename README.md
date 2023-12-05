# insurance-data-analysis
 Insurance Data Analysis - Comprehensive analysis of insurance data, covering diagnostic, predictive, descriptive, and prescriptive analyses.  Key Features :- Diagnostic Analysis - Predictive Modeling - Descriptive Statistics - Prescriptive Analysis  Tech Stack :- Python, Pandas, Scikit-learn, Matplotlib, Jupyter Notebooks


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df=pd.read_csv(r"C:\Users\sk899\Desktop\scripts\insurance.csv.csv",encoding="UTF-8")
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>Gender</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>Male</td>
      <td>27.900</td>
      <td>0</td>
      <td>0</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>Female</td>
      <td>33.770</td>
      <td>1</td>
      <td>1</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>Female</td>
      <td>33.000</td>
      <td>3</td>
      <td>1</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>Female</td>
      <td>22.705</td>
      <td>0</td>
      <td>1</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>Female</td>
      <td>28.880</td>
      <td>0</td>
      <td>1</td>
      <td>3866.85520</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1333</th>
      <td>50</td>
      <td>Female</td>
      <td>30.970</td>
      <td>3</td>
      <td>1</td>
      <td>10600.54830</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>18</td>
      <td>Male</td>
      <td>31.920</td>
      <td>0</td>
      <td>1</td>
      <td>2205.98080</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>18</td>
      <td>Male</td>
      <td>36.850</td>
      <td>0</td>
      <td>1</td>
      <td>1629.83350</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>21</td>
      <td>Male</td>
      <td>25.800</td>
      <td>0</td>
      <td>1</td>
      <td>2007.94500</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>61</td>
      <td>Male</td>
      <td>29.070</td>
      <td>0</td>
      <td>0</td>
      <td>29141.36030</td>
    </tr>
  </tbody>
</table>
<p>1338 rows × 6 columns</p>
</div>




```python
# Descriptive analysis
# charges by age
sns.scatterplot(x=df['age'],y=df['charges'])
```




    <Axes: xlabel='age', ylabel='charges'>




    
![png](output_3_1.png)
    



```python
sns.scatterplot(x=df['children'],y=df['smoker'])
```




    <Axes: xlabel='children', ylabel='smoker'>




    
![png](output_4_1.png)
    



```python
sns.scatterplot(x=df['age'],y=df['smoker'])
```




    <Axes: xlabel='age', ylabel='smoker'>




    
![png](output_5_1.png)
    



```python
df['bmi'].sum()+df['charges'].sum()
```




    17796852.615759




```python
# Dignostic Analysis
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>Gender</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>Male</td>
      <td>27.900</td>
      <td>0</td>
      <td>0</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>Female</td>
      <td>33.770</td>
      <td>1</td>
      <td>1</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>Female</td>
      <td>33.000</td>
      <td>3</td>
      <td>1</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>Female</td>
      <td>22.705</td>
      <td>0</td>
      <td>1</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>Female</td>
      <td>28.880</td>
      <td>0</td>
      <td>1</td>
      <td>3866.85520</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1333</th>
      <td>50</td>
      <td>Female</td>
      <td>30.970</td>
      <td>3</td>
      <td>1</td>
      <td>10600.54830</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>18</td>
      <td>Male</td>
      <td>31.920</td>
      <td>0</td>
      <td>1</td>
      <td>2205.98080</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>18</td>
      <td>Male</td>
      <td>36.850</td>
      <td>0</td>
      <td>1</td>
      <td>1629.83350</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>21</td>
      <td>Male</td>
      <td>25.800</td>
      <td>0</td>
      <td>1</td>
      <td>2007.94500</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>61</td>
      <td>Male</td>
      <td>29.070</td>
      <td>0</td>
      <td>0</td>
      <td>29141.36030</td>
    </tr>
  </tbody>
</table>
<p>1338 rows × 6 columns</p>
</div>




```python
# Predictive Analysis
```


```python
df = df.drop(columns=['Gender'])
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>27.900</td>
      <td>0</td>
      <td>0</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>33.770</td>
      <td>1</td>
      <td>1</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>33.000</td>
      <td>3</td>
      <td>1</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>22.705</td>
      <td>0</td>
      <td>1</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>28.880</td>
      <td>0</td>
      <td>1</td>
      <td>3866.85520</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1333</th>
      <td>50</td>
      <td>30.970</td>
      <td>3</td>
      <td>1</td>
      <td>10600.54830</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>18</td>
      <td>31.920</td>
      <td>0</td>
      <td>1</td>
      <td>2205.98080</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>18</td>
      <td>36.850</td>
      <td>0</td>
      <td>1</td>
      <td>1629.83350</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>21</td>
      <td>25.800</td>
      <td>0</td>
      <td>1</td>
      <td>2007.94500</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>61</td>
      <td>29.070</td>
      <td>0</td>
      <td>0</td>
      <td>29141.36030</td>
    </tr>
  </tbody>
</table>
<p>1338 rows × 5 columns</p>
</div>




```python
X=df.drop(columns=['charges'])
```


```python
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>27.900</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>33.770</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>33.000</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>22.705</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>28.880</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1333</th>
      <td>50</td>
      <td>30.970</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>18</td>
      <td>31.920</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>18</td>
      <td>36.850</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>21</td>
      <td>25.800</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>61</td>
      <td>29.070</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1338 rows × 4 columns</p>
</div>




```python
y=df['charges']
```


```python
y
```




    0       16884.92400
    1        1725.55230
    2        4449.46200
    3       21984.47061
    4        3866.85520
               ...     
    1333    10600.54830
    1334     2205.98080
    1335     1629.83350
    1336     2007.94500
    1337    29141.36030
    Name: charges, Length: 1338, dtype: float64




```python
!pip install scikit-learn
```

    Requirement already satisfied: scikit-learn in c:\users\sk899\appdata\local\programs\new folder\lib\site-packages (1.3.2)
    Requirement already satisfied: numpy<2.0,>=1.17.3 in c:\users\sk899\appdata\local\programs\new folder\lib\site-packages (from scikit-learn) (1.24.3)
    Requirement already satisfied: scipy>=1.5.0 in c:\users\sk899\appdata\local\programs\new folder\lib\site-packages (from scikit-learn) (1.11.1)
    Requirement already satisfied: joblib>=1.1.1 in c:\users\sk899\appdata\local\programs\new folder\lib\site-packages (from scikit-learn) (1.2.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\sk899\appdata\local\programs\new folder\lib\site-packages (from scikit-learn) (2.2.0)
    


```python
from sklearn.model_selection import train_test_split
```


```python
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
```


```python
from sklearn.linear_model import LinearRegression
```


```python
#Define the model
```


```python
mymodel=LinearRegression()
```


```python
# Prescriptive Analysis 
```


```python
mymodel.fit(x_train, y_train)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
test_predictions = mymodel.predict(x_test)
```


```python
# Visualize the predicted charges against the actual charges
plt.scatter(y_test, test_predictions)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual Charges vs Predicted Charges')
plt.show()
```


    
![png](output_25_0.png)
    



```python
# Display the model coefficients
print("Model Coefficients:")
for feature, coefficient in zip(X.columns, mymodel.coef_):
    print(f"{feature}: {coefficient}")
```

    Model Coefficients:
    age: 265.19654688509377
    bmi: 308.77795877831915
    children: 360.52183772061835
    smoker: -23660.384437047363
    


```python
df['predicted_charges'] = mymodel.predict(X)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>charges</th>
      <th>predicted_charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>27.900</td>
      <td>0</td>
      <td>0</td>
      <td>16884.92400</td>
      <td>25414.274540</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>33.770</td>
      <td>1</td>
      <td>1</td>
      <td>1725.55230</td>
      <td>3661.742011</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>33.000</td>
      <td>3</td>
      <td>1</td>
      <td>4449.46200</td>
      <td>6796.992127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>22.705</td>
      <td>0</td>
      <td>1</td>
      <td>21984.47061</td>
      <td>3862.540263</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>28.880</td>
      <td>0</td>
      <td>1</td>
      <td>3866.85520</td>
      <td>5504.047612</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1333</th>
      <td>50</td>
      <td>30.970</td>
      <td>3</td>
      <td>1</td>
      <td>10600.54830</td>
      <td>12004.496903</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>18</td>
      <td>31.920</td>
      <td>0</td>
      <td>1</td>
      <td>2205.98080</td>
      <td>2729.980950</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>18</td>
      <td>36.850</td>
      <td>0</td>
      <td>1</td>
      <td>1629.83350</td>
      <td>4252.256287</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>21</td>
      <td>25.800</td>
      <td>0</td>
      <td>1</td>
      <td>2007.94500</td>
      <td>1635.849483</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>61</td>
      <td>29.070</td>
      <td>0</td>
      <td>0</td>
      <td>29141.36030</td>
      <td>36913.799721</td>
    </tr>
  </tbody>
</table>
<p>1338 rows × 6 columns</p>
</div>




```python

```
