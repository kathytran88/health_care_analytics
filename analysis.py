
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr 
from scipy import stats
from sklearn.decomposition import PCA

df = pd.read_csv('Healthcare-Diabetes.csv') 

rawDataNum = np.genfromtxt('Healthcare-Diabetes.csv', delimiter = ',' , skip_header = 1)

full_data = rawDataNum[ : , 1 : 10]

data = rawDataNum[ : , 1 : 9]

#clean_missing = ~np.isnan(data) 

#input = data[clean_missing , :]

count_missing = df.isnull().sum()
count_missing 

count_dup = df.duplicated().sum()
count_dup 

df.shape

df.head()

des = df.describe()

# Do higher age indicates higher blood pressure?
sns.regplot(x = 'Age', y='BloodPressure' , data = df,
            scatter_kws={
        'facecolor': 'blue',    
        'edgecolor': 'black',    
        'linewidths': 0.8,        
        's': 30,                
        'alpha': 0.9               
    },
    line_kws={
        'color': 'red',
        'linewidth': 2
    }
            ) 
plt.show() 

r1, p1 = pearsonr(df['Age'] , df['BloodPressure'] ) 
print(f'Correlation is {r1} and p-value is {p1}') 

r1b = df['Age'].corr(df['BloodPressure'], method='pearson')
print(f"Pearson r: {r1b:.3f}") 

# => A low r shows a weak linear relationship between Age and BloodPressure 

# Visualize Age 
plt.figure(figsize=(6,7))
plt.boxplot(df['Age'],
            medianprops=dict(color='red', linewidth=2)
            )
plt.ylabel('Age') 
plt.title('Age distribution')
plt.show() 

# Median is 29 years old => threshold to split the data 
age_blood = data[: , [2,7]] 
mask_young = age_blood[: , 1] <= 29
mask_old = age_blood[: , 1] >= 30

age_blood_young = age_blood[mask_young]
age_blood_old = age_blood[mask_old]

blood_old = age_blood_old[: , 0]
blood_young = age_blood_young[: , 0] 

plt.figure()
plt.hist(blood_old, bins=20, edgecolor='black', linewidth=1)
plt.xlabel('Blood Pressure for people 30 years and older')
plt.ylabel('Frequency')
plt.title('Distribution of blood presure for old people')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


plt.figure()
plt.hist(blood_young, bins=20, edgecolor='black', linewidth=1)
plt.xlabel('Blood Pressure for people 29 years and younger')
plt.ylabel('Frequency')
plt.title('Distribution of blood presure for young people')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

blood_old_var = np.var(blood_old)
blood_young_var = np.var(blood_young)

if blood_old_var > blood_young_var:
    print('Old has higher variance') 
else:
    print('Old has lower variance') 
    
blood_old_var
blood_young_var

t2, p2_two_tailed = stats.ttest_ind(blood_old, blood_young, equal_var=False)

# For the hypothesis blood_old > blood_young:
p_one_tailed = p2_two_tailed / 2 if t2 > 0 else 1 - (p2_two_tailed / 2)

print(f"t = {t2:.3f}, one-tailed p = {p_one_tailed:.3f}")

# Do people with diabetes have higher BMI than those who don't?
bmi_diabetes = full_data[: , [5 , 8] ] 
with_diabetes_mask = bmi_diabetes[ : , 1] == 1 
without_diabetes_mask = bmi_diabetes[ : , 1] == 0
bmi_with_diabetes = bmi_diabetes[with_diabetes_mask]
bmi_without_diabetes = bmi_diabetes[without_diabetes_mask]

bmi_1 = bmi_with_diabetes[: , 0]
bmi_0 = bmi_without_diabetes[: , 0] 

plt.figure()
plt.hist(bmi_1, bins=20, edgecolor='black', linewidth=1)
plt.xlabel('BMI for people with diabetes')
plt.ylabel('Frequency')
plt.title('Distribution of people with diabetes')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show() 


plt.figure()
plt.hist(bmi_0, bins=20, edgecolor='black', linewidth=1)
plt.xlabel('BMI for people without diabetes')
plt.ylabel('Frequency')
plt.title('Distribution of people without diabetes')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

bmi_1_var = np.var(bmi_1) 
bmi_0_var = np.var(bmi_0)

t3, p3_two_tailed = stats.ttest_ind(bmi_1, bmi_0, equal_var = False) 
p3_one_tailed = p3_two_tailed / 2 if t3 > 0 else 1 - (p3_two_tailed / 2) 

print(f"t = {t3:.3f}, one-tailed p = {p3_one_tailed:.3f}")

# Divide Age into bins (< 30, 30–40, 40–50, > 50). What’s the diabetes prevalence in each bin?
age_diabetes = full_data[:, [7, 8]]
ages         = age_diabetes[:, 0] 
outcomes     = age_diabetes[:, 1]

mask_21_44 = (ages >= 21) & (ages < 45)
mask_45_64 = (ages >= 45) & (ages < 65)
mask_65_up  =  ages >= 65

group_21_44 = age_diabetes[mask_21_44]
group_45_64 = age_diabetes[mask_45_64]
group_65_up  = age_diabetes[mask_65_up] 

#### Pie Chart ####       

groups = [
    ('21–44 years old', group_21_44),
    ('45–64 years old', group_45_64),
    ('65+ years old',   group_65_up)
]

for label, group in groups:
    outcomes = group[: , 1] 
    counts = [ np.sum(outcomes == 0),
              np.sum(outcomes == 1)
            ]
    
    plt.figure()
    plt.pie(
        counts,
        labels=['Without Diabetes', 'With Diabetes'],
        autopct='%1.1f%%'
    )
    plt.title(f'Diabetes Prevalence ({label})') 
    plt.show()


# What’s the Pearson correlation between Glucose and Insulin?
sns.regplot(x='Insulin', y='Glucose', data=df , 
            scatter_kws={
        'facecolor': 'blue',    
        'edgecolor': 'black',    
        'linewidths': 0.8,        
        's': 30,                
        'alpha': 0.9               
    },
    line_kws={
        'color': 'red',
        'linewidth': 2
    }
            ) 
plt.show() 

r, pvalue = pearsonr(df['Insulin'] , df['Glucose'])
print(f'The correlation is {r} and the p value is {pvalue}')

# Build predictive model to predict diabetes

# Detect multicollinearity
predictors = df.loc[: , 'Pregnancies': 'Age']

corr_matrix = predictors.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,        
    fmt=".2f",        
    cmap="coolwarm",   
    square=True
)
plt.title("Predictor Correlation Matrix")
plt.show()

## PCA 
zscoredData = stats.zscore(data)
pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_

numPredictors = 8
plt.bar(np.linspace(1,numPredictors,numPredictors),eigVals)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.plot([0,numPredictors],[1,1],color='orange', linestyle='--')
plt.show()

loadings = pca.components_
origDataNewCoordinates = pca.fit_transform(zscoredData)

varExplained = eigVals/sum(eigVals)*100

# Now let's display this for each factor:
count = 0
for ii in range(len(varExplained)):
    count += 1
    print('Variance explained by factor ' + str(count) + ' is ' + str(varExplained[ii].round(3)) + '%') 

total = 0 
for eigen in range(len(varExplained)):
    total += varExplained[eigen] 
    print(eigen)
    
    if total > 90:
        print(total) 
        break

'''
plt.subplot(1,2,1) # Factor 1: 
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:]) 
plt.title('Name1')
plt.subplot(1,2,2) # Factor 2:
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:])
plt.title('Name2')
plt.show()
'''






  








