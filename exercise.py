print('Hello World')
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import math as m
# Data
x = np.array([139.0, 138.0, 139.0, 120.5, 149.0, 141.0, 141.0, 150.0, 166.0, 151.5, 129.5, 150.0])
y = np.array([110, 60, 90, 60, 85, 100, 95, 85, 155, 140, 105, 110])

# Variablen
r = np.corrcoef(x,y)[0,1]
n = np.size(x)
x_mean,y_mean = x.mean(), y.mean()
b1 = r * y.std(ddof=1)/x.std(ddof=1)
b0 = y_mean - b1 * x_mean
y_pred = b1 * x + b0

print("De Pearson Coefficient(r) = ", r)
print("b1 = ", b1, " b0= ", b0)

# Rekenen met variablen
se = np.sqrt(np.sum((y-y_pred)**2)/(n-2))
sb1 = se/np.sqrt(np.sum((x-x_mean)**2))

print("Standard error = ", se)
print("Standard deviation of the slope(sb1) = ", sb1)


# Normaliteit checken
residuals = y - y_pred
st.probplot(residuals,plot =plt)
plt.plot()

# Hypothesis verwerpen of niet (2-tailed)
alpha = 0.05
t_half_alpha = st.t.ppf(1-alpha/2,df=n-2)
t0 = r * m.sqrt(n-2)/m.sqrt(1-r**2)
p_value = 2 * st.t(df=n-2).sf(abs(t0))
print("T-test= t0:",t0, "vs t_half_alpha:", t_half_alpha)
print("P_val test= p_value:",p_value, "vs alpha:", alpha)

# Confidence interval (margin of error)
moe = t_half_alpha * sb1
LB = b1 - moe
UB = b1 + moe
print("LB:",LB, "UB:", UB)

# Sterkte x=3000 psi?
x_new = 146
y_pred = b0 + b1 * x_new

print(f"Het gewicht bij 146 cm is {y_pred}")