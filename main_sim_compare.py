#import display.line_graph as line_graph
import streamlit as st; st.set_page_config(layout="wide")
import altair as alt
#import display.hide_streamlit #as st_hide
import streamlit as st; #st.set_page_config(layout="wide")
#from streamlit_option_menu import option_menu
import numpy as np 
import plotly.express as px
import pandas as pd; #import get_data
#import matplotlib.pyplot as plt
#from numpy.random import default_rng
#RNG = default_rng().standard_normal
import process as proc
import get as get
import display as disp

MAX_LENGTH = 2000
DECIMALS = 0
# MONEY_MULTIPLIER = 1000 In process file.
N = 4
NN = {}; _ = [NN.update({ n: n }) for n in range (N)]

# compliance probability (investing the same amount in a non-house-buying situation)
# interés variable
# mc sim on: interest, ingresos, occupancy, stock market

# problem with (i+1) vs (i+i) in some series

Columns, colGraph = proc.get_columns(N)
inputs = get.read_data()[0:N]
processed = [{} for i in range(N)]

readdata = get.read_data()[0:N]
colnames = [inputs[i]['name'] for i in range(N)]
#ingresos = [MONEY_MULTIPLIER*inputs[i]['ingreso_pesimista'] for i in range(N)]
inversiones = [inputs[i]['inversion'] for i in range(N)]
tasas = [inputs[i]['tasa'] for i in range(N)]
#porciones = [inputs[i]['porcion'] for i in range(N)]
#meses_con_prestamos = [inputs[i]['meses_con_prestamo'] for i in range(N)]
retrasos = [inputs[i]['retraso'] for i in range(N)]
valorizaciones = [inputs[i]['valorizacion'] for i in range(N)]
max_desembolso_mensual = inputs[0]['max_desembolso_mensual'] # max_desembolso_mensual = [inputs[n]['max_desembolso_mensual'] for n in range(N)]

# Names
for i in range(N):
	with Columns[i]:
		key = "Sim" + str(i)
		inputs[i].update({ 'name': st.text_input('', value=inputs[i]['name'], key=key)})
		hide_graph = st.checkbox("", value=inputs[i]['hide_graph'], key=f'hide_graph_{key}', on_change=None, disabled=False)
		inputs[i].update({ 'hide_graph': hide_graph })

# Amounts
with st.sidebar:
	st.subheader("SM Invs.  \n  ")
	expanders = [st.expander(label=f"Inv: {inputs[i]['name']}") for i in range(N)]
	for i in range(N):
		with expanders[i]:
			inputs[i].update(proc.input_sidebar(inputs[i]))

	st.markdown("""---""")
	max_desembolso_mensual = st.number_input('Max desembolso mensual', 0, 30, max_desembolso_mensual)
	_ = [inputs[n].update({ 'max_desembolso_mensual': max_desembolso_mensual}) for n in range(N)]
	meses_display = st.selectbox('Ilustración (meses)', np.arange(60, 721, 60), index=4, key="meses_display")

	st.markdown("""---""")
	r_mes_opor = st.slider("Crec. opor. alt.", 0, 15, 0, 1)
	r_mes_opor = 1 + (r_mes_opor/100)
	r_mes_opor = np.e**(np.log(r_mes_opor)/12)

for i in range(N):
	with Columns[i]:
		inputs[i].update(proc.input_cols(inputs[i]))


# Do some math
loan_matrix = proc.loan_matrix(inputs, MAX_LENGTH) 
xs_repay, ys_repay = proc.calc_min_repay_times(inputs, loan_matrix)
del loan_matrix
processed = proc.add_dict_to_processed(processed, 'loan_repay_time', xs_repay)
processed = proc.add_dict_to_processed(processed, 'loan_repay_amount', ys_repay)

loans = proc.loans(inputs, processed)
processed = proc.add_list_to_processed(processed, 'loan', loans)

# Price of loans. # If loan/cap_ini is very high, then ys_repay will be outside MAX_LENGTH
y = proc.prices_of_loans(inputs, processed)
processed = proc.add_list_to_processed(processed, 'costo_prestamo', y)
print("Costo/price: ", processed[0])

y = proc.cash_and_capital_spent(inputs, processed)
processed = proc.add_list_to_processed(processed, 'cash_and_capital_spent', y)

_ = [processed[n].update({'deuda_y_capital': inputs[n]['inversion'] + processed[n]['costo_prestamo']}) for n in range(N)]

y = proc.cash_spent_during_loan_repayment(inputs, processed)
processed = proc.add_list_to_processed(processed, 'cash_spent_during_loan_repayment', y)
y = proc.cash_and_income_spent_during_loan_repayment(inputs, processed)
processed = proc.add_list_to_processed(processed, 'cash_and_income_spent_during_loan_repayment', y)

# Retraso sólo opera en ingresos; igual hay valorización durante la etapa de construcción.
# Curva de valorización colombiana

# Matrices
income_matrix = proc.income_matrix(inputs, MAX_LENGTH)
growth_matrix = proc.growth_matrix(inputs, MAX_LENGTH) # Valorización también sobre planos.
desembolso_matrix = proc.desembolso_matrix(inputs, MAX_LENGTH)

shifts = [inputs[n]['retraso'] for n in NN]
D_matrix = proc.shift_sequences(desembolso_matrix, shifts)

zeros = [0 for _ in NN]
#debt_matrix = proc.debt_matrix(inputs, processed, MAX_LENGTH)

# Shift all matrices one step to the right?

# Investments
mmatrix = income_matrix + growth_matrix
temp = mmatrix[0:N, 0:meses_display].view()
y_m = temp[:,-1].copy()
y_m = proc.to_dict(y_m, int_=True)
x_m = proc.x_intercepts_for_y(mmatrix, targets=y_m)

# Saldar
smatrix = income_matrix[::] + D_matrix[::]
y_s = [processed[n]['loan_repay_amount'] for n in NN]
smatrix = proc.shift_down(smatrix, y_s)
x_s = proc.x_intercepts_for_y(smatrix, targets=zeros)
smatrix = proc.cut_after(smatrix, targets=zeros)

# Restaurar cap_ini para nueva compra
rmatrix0 = income_matrix[:] + D_matrix[::] # + desembolso_matrix[:]
rmatrix1 = income_matrix[:] + D_matrix[::] # + desembolso_matrix[:]
y_r = [inputs[n]['cap_ini'] for n in NN]
x_r = proc.x_intercepts_for_y(rmatrix1, targets=y_r)

duo_matrix = np.zeros([N*2, MAX_LENGTH])
duo_matrix[0:N] = rmatrix0
duo_matrix[N:N*2] = rmatrix1
steps = y_s
steps.extend(y_s)
duo_matrix = proc.shift_down(duo_matrix, steps)
y_duo = [inputs[n]['cap_ini'] for n in NN]
y_duo.extend(y_duo)
x_duo = proc.x_intercepts_for_y(duo_matrix, targets=y_duo)
duo_matrix[0:N] = proc.cut_after(duo_matrix[0:N], targets=zeros)
duo_matrix[N:N*2] = proc.cut_before(duo_matrix[N:N*2], targets=zeros)
duo_matrix[N:N*2] = proc.cut_after(duo_matrix[N:N*2], targets=y_duo)

# Se paga solo, inclusive valorización
deudasycaps = np.array([processed[n]['deuda_y_capital'] for n in NN]).reshape(N, 1)
pmatrix = income_matrix[:] + growth_matrix[:] - deudasycaps
y_p = [inputs[n]['inversion'] for n in NN]
x_p = proc.x_intercepts_for_y(pmatrix, targets=y_p)


# Remember that intercept 0 is the first month, intercept 30 is month 31, etc.

# Saldar+RecCap

# For display, remember to shift everything one step, so that month 0 becomes 1.

#print(smatrix[:1,0:42]); exit()


# Crop matrix
mmatrix = mmatrix[0:N,0:meses_display+1]
pmatrix = pmatrix[0:N,0:meses_display+1]
#smatrix = smatrix[0:N,0:meses_display+1]

smatrix = smatrix[0:N,0:proc.len_longest_graph(smatrix)+11]
duo_matrix = duo_matrix[0:N*2,0:proc.len_longest_graph(duo_matrix)+11]


# Opportunity-cost graphs
for l in range(N):
	break
	line_data = cost_opor_matrix[l]
	for i in range(line_data.shape[0]):
		d = { 'name': cost_opor_labels[l], 'mes': i, 'y': line_data[i] }
		#data.append(d)
		if r_mes_opor < 1.00001:
			pass
		else:
			data.append(d)


#columns = [colA, colB, colC, colD]
for i in range(len(Columns)):
	with Columns[i]:
		m = meses_display-1
		m = meses_display
		#crecimiento_anual_alt = np.e**(np.log(mmatrix[i][m]/(inputs[i]['deuda_y_capital']))/(m/12))
		#crecimiento_anual_alt = np.e**(np.log(mmatrix[i][m]/(processed[i]['deuda_y_capital']))/(m/12))
#		crecimiento_anual_alt = np.e**(np.log(mmatrix[:,-1][i]/(processed[i]['cash_and_capital_spent']))/(m/12))
		costosprestamos = [round(processed[_]['costo_prestamo'], DECIMALS) for _ in NN]
		st.write("Costo", costosprestamos[i])
		st.write("$ gastado", processed[i]['cash_and_capital_spent'])
#		st.write(round(crecimiento_anual_alt, 4)) #st.write(round(mmatrix[0][m]), "M")
		st.write(xs_repay[i], " meses")
		#st.write(round(-processed[i]['deuda_y_capital']))

# Save/restore
with Columns[0]:
	if st.button("Save"):
		data_to_file = pd.DataFrame(inputs)
		data_to_file.to_csv("save.csv", index=True)

with colGraph:
	tab1, Banco, BancoCap, PagaSolo = st.tabs(['Inv  ', '| Banco ', '| Banco+Cap ','| Paga solo ']) #PagaSolo
	with tab1:
		costos = [processed[n]['costo_prestamo'] for n in range(N)]
		xs_inv, ys_inv = proc.find_x_intercepts_for_y(mmatrix, targets=costos)

		disp.plot(inputs, mmatrix, x_m, y_m, show_labels=True, labels=y_m)

	with Banco:
		disp.plot(inputs, smatrix, x_s, zeros, labels=x_s)
		st.write("Tiempo requerido para saldar cuentas con el banco.")

	with BancoCap:
		disp.plot_duo(inputs, duo_matrix, x_duo, y_duo, labels=x_duo)
		st.write("Tiempo para volver a tener el mismo capital inicial.  \nIncluye ahorros (desembolso) mensuales.")

	with PagaSolo:
		###disp.plot(inputs, mmatrix_shifted_down, x2s, y2s)
		#disp.plot(inputs, incomes_only_shifted_inverse, xs_recovercap, ys_recovercap)

		#disp.plot(inputs, rmatrix0, x_rc0, y_rc1, labels=x_rc0)

		pmatrix = pmatrix[:,:]
		#pmatrix = pmatrix[0:N,0:proc.len_longest_graph(pmatrix)+11]
		disp.plot(inputs, pmatrix, x_p, y_p, labels=x_p)
		st.write("-Costo de préstamo + valor y producción de la inversión.")
#st.write(processed[0:2])
	# Graph debt remaining