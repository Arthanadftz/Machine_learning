import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
import ml_metrics as metrics

df = pd.read_csv('364d9c85Inp.csv', ';', index_col=['week'], parse_dates=['week'], dayfirst=True)

sales = df.qty

# Строим график проданных товаров за временной интервал
"""graph = plt.figure(figsize=(12, 6))
ax = plt.subplot()
plt.plot(sales)
plt.title('Sales per week')
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df.index, fontsize=5, rotation=90)
ax.tick_params(width=2)
plt.show()"""

# Выводим информацию о статистических показателях ряда
itog = sales.describe()
print(itog)

# Коэффициент вариации
print('V = %f' %(itog['std'] / itog['mean']))

# Tест Харки — Бера
row =  [u'JB', u'p-value', u'skew', u'kurtosis']
jb_test = sm.stats.stattools.jarque_bera(sales)
a = np.vstack([jb_test])
itog = SimpleTable(a, row)
print(itog)

# Тест Дикки-Фуллера
test = sm.tsa.adfuller(sales)
print('adf: ', test[0]) 
print('p-value: ', test[1])
print('Critical values: ', test[4])
if test[0] > test[4]['5%']: 
    print('Есть единичные корни, ряд не стационарен')
else:
    print('Единичных корней нет, ряд стационарен')

# Определение порядка интегрирования ряда
sales1diff = sales.diff(periods=1).dropna()

test = sm.tsa.adfuller(sales1diff)
print('adf: ', test[0])
print('p-value: ', test[1])
print('Critical values: ', test[4])
if test[0]> test[4]['5%']: 
    print('Есть единичные корни, ряд не стационарен')
else:
    print('Единичных корней нет, ряд стационарен')

# Разбивка ряда на промежутки для проверки мат. ожидания на интервалах
m = sales1diff.index[len(sales1diff.index)//2+1]
r1 = sm.stats.DescrStatsW(sales1diff[m:])
r2 = sm.stats.DescrStatsW(sales1diff[:m])
print('p-value: ', sm.stats.CompareMeans(r1,r2).ttest_ind()[1])

# Проверка на гипотезы стационарности ряда, исходя из высокого значения p-value
"""graph = plt.figure(figsize=(12, 6))
ax = plt.subplot()
plt.plot(sales1diff, c='red', lw=2)
plt.title('Sales per week')
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df.index, fontsize=5, rotation=90)
ax.tick_params(width=2)
plt.show()"""

# Тренд отсутствует, таким образом ряд первых разностей является стационарным, а исходный ряд — интегрированным рядом первого порядка.

""" Для построения модели ARIMA необходимо определить параметры p — порядок компоненты AR, d — порядок интегрированного ряда и q — порядок компонетны MA.
	Параметр d - определен и равен 1(line-77), остальные определим по коррелограммам ACF(Автокорреляционная ф-я) и PACF(Частично а-к ф-я).
"""

# Строим графики ACF и PACF
"""fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sales1diff.values.squeeze(), lags=25, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sales1diff, lags=25, ax=ax2)

plt.show()"""

# Определяем по диаграммам, что p = q = 1, т.к. присутствует 1 лаг, сильно отличный от нуля.

# Строим модель ARIMA

src_data_model = sales[:'2018-01']
model = sm.tsa.ARIMA(src_data_model, order=(1,1,1)).fit(full_output=True, disp=0)

print(model.summary())

# Q-тест Льюнга — Бокса для проверки гипотезы о том, что остатки случайны, т.е. являются «белым шумом».
q_test = sm.tsa.stattools.acf(model.resid, qstat=True) #свойство resid, хранит остатки модели, qstat=True, означает что применяем указынный тест к коэф-ам
print(pd.DataFrame({'Q-stat':q_test[1], 'p-value':q_test[2]}))
""" Значение данной статистики и p-values, свидетельствуют о том, 
	что гипотеза о случайности остатков не отвергается, 
	и скорее всего данный процесс представляет «белый шум».
"""
# Предсказания модели
pred = model.predict(155, 158, typ='levels')
trn = sales['2018-01':'2018-04']
print(pred)
print(trn)
# Выводим оценки прогноза модели
# Pасчитаем коэффициент детерминации R^2, чтобы понять какой процент наблюдений описывает данная модель
r2 = r2_score(trn, pred[:4])
print('R^2: %1.2f' % r2)
# Выводим средне-квадр. отклонение и абсолютную ошибку прогноза
print('RMSE %1.2r' %(metrics.rmse(trn,pred[:4])))
print('MAE: %1.2f' %(metrics.mae(trn,pred[:4])))

# Строим график прогноза модели на кривой реальных продаж
graph = plt.figure(figsize=(12, 6))
ax = plt.subplot()
plt.plot(sales, c='green', lw=3, label='Actual')
plt.plot(pred, c='red', lw=3, label='Jan predictions')
plt.title('Sales per week')
plt.legend()
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df.index, fontsize=5, rotation=90)
ax.tick_params(width=2)

plt.show()

