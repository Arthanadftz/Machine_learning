import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Загружаем данные из csv файла в переменную при помощи pandas
df = pd.read_csv('demand_planning_exercice_1_5.csv', ';', index_col=['week'], parse_dates=['week'])

# Создаем список только уникальных sku
sku_u = df.sku.unique()
# Создаем 4 списка для заполнения результирующего файла
sku_w = []
for i in range(len(sku_u)):
	for j in range(4):
		sku_w.append(sku_u[i])

qty_jan = [] 
dpa_test = []
dpa_jan = []

# Функция для вычисления оценки DPA
def get_dpa(y_pred, y_true):
	diff = 0
	for i in range(len(y_pred)):
		diff += y_pred[i] - y_true[i]
	return 1 - abs(diff)/abs(sum(y_pred))


def predict_jan(sku_id):
	# Выбираем строки для обучения модели по каждому товару
	sales_i = df[(df.sku == sku_id)]
	sales_i = sales_i[:'2017-50']
	# Выбираем строки для прогноза на январь
	sales_jan = df[(df.sku == sku_id)]
	sales_jan = sales_jan['2018-01':'2018-04']
	
	sales = sales_i.qty
	sales_jan_qty = sales_jan.qty
	
	# Создаем numpy массивы для обучения и прогноза модели
	X = np.array(range(len(sales)))
	X = X.reshape(-1, 1)
	X_jan = np.array(range(len(sales), len(sales)+4))
	X_jan = X_jan.reshape(-1, 1)
	
	# Разделяем данные на тренировачный и тестовый набор
	X_train, X_test, y_train, y_test = train_test_split(X, sales, test_size=0.15, random_state=42)

	# Создаем модель, обучаем ее на тренировочном наборе данных, делаем прогноз на тестовом наборе и на январь 
	model = LinearRegression()
	model.fit(X_train, y_train)
	sales_pred = model.predict(X_test)
	sales_pred_jan = model.predict(X_jan)

	# Заполняем полученными данными список qty_jan
	for i in sales_pred_jan:
		qty_jan.append(i)

	# Оцениваем модель и добавляем результаты в списки dpa_test и dpa_jan
	dpa_jan.append(get_dpa(sales_pred_jan, sales_jan_qty))
	dpa_test.append(get_dpa(sales_pred, y_test))

# Вызываем функцию для каждого товара	
for i in sku_u:
	predict_jan(i)

# Создаем результирующий DataFrame, содержащий прогнозы на 4 недели января по каждому товару
result = pd.DataFrame({'sku' : sku_w, 'qty': qty_jan})
result = result.set_index(['sku'])
# Создаем результирующий DataFrame, содержащий sku уникальных товаров и соотв. оценку прогноза
dpa_jan_res = pd.DataFrame({'sku': sku_u, 'dpa': dpa_jan})
dpa_jan_res = dpa_jan_res.set_index(['sku'])

# Записываем полученные DataFrameв csv файлы
result.to_csv('result.csv')
dpa_jan_res.to_csv('dpa_jan.csv')

# Оценку на тестовых данных запишем в txt файл
with open('dpa.txt', 'w') as file:
    for i in dpa_test:
    	file.write(str(i) + '\n')