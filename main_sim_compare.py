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
disp.streamlit_hide(st.markdown)

STRANGE_CONSTANT = 0.0000001
MAX_LENGTH = 2000
DECIMALS = 0
# MONEY_MULTIPLIER = 1000 In process file.
N = 4
N_opor = 1
N_tot = N + N_opor
NN = {}; _ = [NN.update({ n: n }) for n in range (N)]

curva_de_aplanamiento = [8, 7, 6, 5, 5, 5, 5, 5, 5, 4]

# compliance probability (investing the same amount in a non-house-buying situation)
# interés variable
# mc sim on: interest, ingresos, occupancy, stock market

# problem with (i+1) vs (i+i) in some series

# Chain SimA -> SimB

# Curva de aplanamiento

#SuperLeft, SuperRight = st.columns([1, 1])
#ColExpander = st.expander('Variables', expanded=False)
Columns, colGraph = proc.get_columns(N)
Columns2, colGraph2 = proc.get_columns(N)
#ColContainer = st.container()
#Expander = st.expander('Vars')
#ContHideButton = st.container()
sidebar = st.sidebar
#Columns2, _ = proc.get_columns(N)
#SaveRestoreContainer = st.container()
# Save/restore
file = "save.csv"
#with SaveRestoreContainer: #with Columns[0]:
#	if st.button("Restore"):
#		file = "save.csv"
#	if st.button("Restore II"):
#		file = "save2.csv"

inputs = get.read_data(file)[0:N]
processed = [{} for i in range(N)]

#readdata = get.read_data()[0:N]
colnames = [inputs[i]['name'] for i in range(N)]
#ingresos = [MONEY_MULTIPLIER*inputs[i]['ingreso_pesimista'] for i in range(N)]
inversiones = [inputs[i]['inversion'] for i in range(N)]
tasas = [inputs[i]['tasa'] for i in range(N)]
#porciones = [inputs[i]['porcion'] for i in range(N)]
#meses_con_prestamos = [inputs[i]['meses_con_prestamo'] for i in range(N)]
retrasos = [inputs[i]['retraso'] for i in range(N)]
valorizaciones = [inputs[i]['valorizacion'] for i in range(N)]
max_desembolsos = [inputs[i]['max_desembolso_mensual'] for i in NN] # max_desembolso_mensual = [inputs[n]['max_desembolso_mensual'] for n in range(N)]
#st.write("File: ", [inputs[n]['max_desembolso_mensual'] for n in NN])

for i in range(N):
	with Columns2[i]:
		key = "Sim" + str(i)
		#invisibles = [inputs[n][hide_graph] for n in NN]
		invisible = st.checkbox("", value=inputs[i]['hide_graph'], key=f'hide_graph_{key}', on_change=None, disabled=False)
		#if invisible == True:
		#	hide_graph = False
		#elif invisible == False:
		#	hide_graph = True
		inputs[i].update({ 'hide_graph': invisible })

with Columns2[0]:
	chain = st.checkbox("Cadena!", value=False, key='cadena', on_change=None, disabled=False)

for i in range(N):
	with Columns[i]:
		key = "Sim" + str(i)
		inputs[i].update({ 'name': st.text_input('', value=inputs[i]['name'], key=key, disabled=inputs[i]['hide_graph'])})
		inputs[i].update(proc.input_cols(inputs[i], chain))

with sidebar:
	st.subheader("SM Invs.  \n  ")
	expanders = [st.expander(label=f"Inv: {inputs[i]['name']}") for i in range(N)]
	for i in range(N):
		with expanders[i]:
			inputs[i].update(proc.input_sidebar(inputs[i]))

	#_ = [inputs[n].update({ 'max_desembolso_mensual': st.number_input('Max desembolso mensual', 0, 30, max_desembolsos[n], key=f'maxdes_{inputs[n]["name"]}')}) for n in NN]
	#st.write("BBB ", [inputs[n]['max_desembolso_mensual'] for n in NN])
	opor_expander = st.expander(label="Max desembolso mensual", expanded=False)
	with opor_expander:
		if chain == True:
			number = st.number_input('', 0.0, 30.0, value=max_desembolsos[0], key=f'maxdes_{inputs[0]["name"]}')
			if  number == 0 or number == None:
				number = STRANGE_CONSTANT
			suma = number
			processed[0].update({ 'max_desembolso_mensual': suma })
			st.write(f'{inputs[0]["name"]}', suma)
			for n in range(1, N):
				suma = suma + inputs[n-1]['ingreso_pesimista']
				processed[n].update({ 'max_desembolso_mensual': suma })
				st.write(f'{inputs[n]["name"]}', suma)
			# (n-(n-1))*
				#number = st.number_input(f'{inputs[n]["name"]}', 0.0, 30.0, value=max_desembolsos[n]+inputs[n]['ingreso_pesimista'], key=f'maxdes_{inputs[n]["name"]}')
				#number = st.number_input('Tasa bancaria', value=inn['tasa']*100, key=f'tasa_{name}', disabled=disabled) / 100
		else:
			for n in NN:
				max_ = st.number_input(f'{inputs[n]["name"]}', 0.0, 30.0, value=max_desembolsos[n], key=f'max_des_{n}')
				inputs[n].update({ 'max_desembolso_mensual': max_ })

	#st.write("CCC ", [inputs[n]['max_desembolso_mensual'] for n in NN])
	#_ = [inputs[n].update({ 'max_desembolso_mensual': max_desembolso_mensual}) for n in NN]
	#max_desembolso_mensual = st.number_input('Max desembolso mensual', 0, 30, max_desembolso_mensual)

	#_ = [inputs[n].update({ 'max_desembolso_mensual': max_desembolso_mensual}) for n in range(N)]
	#meses_display = st.selectbox('Ilustración (meses)', np.arange(60, 721, 60), index=4, key="meses_display")
	meses_display = 480

	st.markdown("""---""")
	opor_cap_ini = st.number_input('Inversión alt. cap.', 0.0, 1000.0, key='opor_cap_ini')
	#opor_cap_ini = st.number_input('Inversión alt.', 0, 100, key='opor_cap_ini')
	opor_saving = st.number_input('Inversión alt.', 0, 100, key='opor_saving')
	r_mes_opor = st.slider("Crec. opor. alt.", -5, 10, 0, 1)
	r_mes_opor = 1 + (r_mes_opor/100)
	r_mes_opor = np.e**(np.log(r_mes_opor)/12)
	opor_until = st.slider("Opor.: Meses", 0, 240, 120, 12)

	
# Do some math
loan_matrix = proc.loan_matrix(inputs, processed, MAX_LENGTH, chain=chain) 
xs_repay, ys_repay = proc.calc_min_repay_times(inputs, processed, loan_matrix, chain=chain)
del loan_matrix
processed = proc.add_dict_to_processed(processed, 'loan_repay_time', xs_repay)
processed = proc.add_dict_to_processed(processed, 'loan_repay_amount', ys_repay)

loans = proc.loans(inputs, processed, chain=chain)
processed = proc.add_list_to_processed(processed, 'loan', loans)
#st.write(loans)

# Price of loans. # If loan/cap_ini is very high, then ys_repay will be outside MAX_LENGTH
y = proc.prices_of_loans(inputs, processed, chain=chain)
processed = proc.add_list_to_processed(processed, 'costo_prestamo', y)
#print("Costo/price: ", processed[0])

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
income = income_matrix.copy()
#growth = growth_matrix.copy()
desem  = desembolso_matrix.copy()
D_     = proc.shift_sequences(desembolso_matrix, shifts=[inputs[n]['retraso'] for n in NN])
del desem

#D_matrix = proc.shift_sequences(desembolso_matrix, shifts=[inputs[n]['retraso'] for n in NN])
#rmatrix1 = income_matrix[:] + D_matrix[::] # + desembolso_matrix[:]
#y_r = [inputs[n]['cap_ini'] for n in NN]
#x_r = proc.x_intercepts_for_y(rmatrix1, targets=y_r)

growth_matrix = proc.shift_sequences(growth_matrix, shifts=[inputs[n]['shift'] for n in NN])
income_matrix = proc.shift_sequences(income_matrix, shifts=[inputs[n]['shift'] for n in NN])
desembolso_matrix = proc.shift_sequences(desembolso_matrix, shifts=[inputs[n]['shift'] for n in NN])

#opor_seq = proc.opor_sequence(inputs, MAX_LENGTH)[0]
D_matrix = proc.shift_sequences(desembolso_matrix, shifts=[inputs[n]['retraso'] for n in NN])
#D_matrix = proc.shift_right(desembolso_matrix, [inputs[n]['retraso'] for n in NN])

zeros = [0 for _ in NN]
#debt_matrix = proc.debt_matrix(inputs, processed, MAX_LENGTH)

# Shift all matrices one step to the right?

# Investments
mmatrix = np.zeros([N+N_opor, MAX_LENGTH]) # +1: Last row is Oportunidad
mmatrix[0:N] = income_matrix + growth_matrix
temp = mmatrix[0:N, 0:meses_display].view()
y_m = temp[:,-1].copy()
y_m = proc.to_dict(y_m, int_=True)
x_m = proc.x_intercepts_for_y(mmatrix[0:N], targets=y_m)
#print(mmatrix.shape[0], mmatrix.shape[1]); 
#print(opor_matrix.shape[0], opor_matrix.shape[1])
#mmatrix[4:5,0:MAX_LENGTH] = opor_matrix[0]
#print(opor_seq); exit()
#print(mmatrix[4])

# Saldar
smatrix = income_matrix[::] + D_matrix[::]
y_s = [processed[n]['loan_repay_amount'] for n in NN]
smatrix = proc.shift_down(smatrix, y_s)
x_s = proc.x_intercepts_for_y(smatrix, targets=zeros)
smatrix = proc.cut_after(smatrix, targets=zeros)

# Restaurar cap_ini para nueva compra
duo_matrix, x_duo, y_duo = proc.duo_matrix(inputs, N, NN, income_matrix, D_matrix, y_s, MAX_LENGTH)

# Make chain (next graph starts where last finished).
if chain == True:
	duo_matrix, x_duo, y_duo = proc.duo_matrix(inputs, N, NN, income, D_, y_s, MAX_LENGTH)
	cadena = [0]
	xlist = list(x_duo)
	x = xlist[3] # The last or fourth?       BBBBBBBBBBBBBBAD DESIGN using a constant!
	for i in range(0, N-0):
		sss = cadena[i] + x_duo[i]
		cadena.append(sss)
	cadena.pop(-1)

	x_duo = {}
	for i in range(1, len(cadena)): # or len(x_duo)
		x_duo.update({ i-1: cadena[i] })
	x_duo.update({ N-1: sss })
 
	duo_matrix[0:N] = proc.shift_sequences(duo_matrix[0:N], shifts=cadena)
	duo_matrix[N:N*2] = proc.shift_sequences(duo_matrix[N:N*2], shifts=cadena)
	for n in NN:
		inputs[n].update({ 'shift': cadena[n] })

	# Smatrix?
	smatrix = income[::] + D_[::]
	y_s = [processed[n]['loan_repay_amount'] for n in NN]
	smatrix = proc.shift_sequences	(smatrix, shifts=cadena)
	smatrix = proc.shift_down(smatrix, y_s)
	x_s = proc.x_intercepts_for_y(smatrix, targets=zeros)
	smatrix = proc.cut_after(smatrix, targets=zeros)

	#mmatrix = proc.shift_sequences(mmatrix, shifts=cadena)
	#x_m = proc.x_intercepts_for_y(mmatrix[0:N], targets=y_m)




# Se paga solo, inclusive valorización
deudasycaps = np.array([processed[n]['deuda_y_capital'] for n in NN]).reshape(N, 1)
pmatrix = income_matrix[:] + growth_matrix[:] - deudasycaps
y_p = [inputs[n]['inversion'] for n in NN]
x_p = proc.x_intercepts_for_y(pmatrix, targets=y_p)
#st.write("Paga solo 0: ", x_duo)

# Oportunidad alternativa (bolsa)
#opor_matrix = proc.opor_sequence(inputs, r_mes_opor, MAX_LENGTH, untils=x_duo) # opor_sequence2 involves eternal saving; oporsequence until recovered cap_ini
#opor_matrix = proc.opor_sequence2(inputs, r_mes_opor, MAX_LENGTH, until=opor_until) # opor_sequence2 involves eternal saving; oporsequence until recovered cap_ini
opor_matrix = proc.opor_sequence3(opor_cap_ini, opor_saving, r_mes_opor, MAX_LENGTH, until=opor_until) # opor_sequence2 involves eternal saving; oporsequence until recovered cap_ini
opor_matrix = proc.shift_sequences(opor_matrix, shifts=[inputs[n]['shift'] for n in NN])
mmatrix[4] = opor_matrix[0]
#mmatrix[5] = opor_matrix[1]
#mmatrix[6] = opor_matrix[2]
#mmatrix[7] = opor_matrix[3]


# Remember that intercept 0 is the first month, intercept 30 is month 31, etc.

# Saldar+RecCap

# For display, remember to shift everything one step, so that month 0 becomes 1.

#print(smatrix[:1,0:42]); exit()


# Crop matrix
mmatrix = mmatrix[0:N_tot,0:meses_display+1]
pmatrix = pmatrix[0:N,0:max(x_p.values())+6]
#smatrix = smatrix[0:N,0:meses_display+1]

smatrix = smatrix[0:N,0:proc.len_longest_graph(smatrix)+11]
duo_matrix = duo_matrix[0:N*2,0:proc.len_longest_graph(duo_matrix)+11]



#columns = [colA, colB, colC, colD]
#for i in range(len(Columns)):
#	with Columns[i]:
#		m = meses_display-1
#		m = meses_display
		#crecimiento_anual_alt = np.e**(np.log(mmatrix[i][m]/(inputs[i]['deuda_y_capital']))/(m/12))
		#crecimiento_anual_alt = np.e**(np.log(mmatrix[i][m]/(processed[i]['deuda_y_capital']))/(m/12))
#		crecimiento_anual_alt = np.e**(np.log(mmatrix[:,-1][i]/(processed[i]['cash_and_capital_spent']))/(m/12))
#		costosprestamos = [round(processed[_]['costo_prestamo'], DECIMALS) for _ in NN]
		###st.write("Costo", costosprestamos[i])
		###st.write("$ gastado", processed[i]['cash_and_capital_spent'])
#		st.write(round(crecimiento_anual_alt, 4)) #st.write(round(mmatrix[0][m]), "M")
		###st.write(xs_repay[i], " meses")
		#st.write(round(-processed[i]['deuda_y_capital']))

#with SaveRestoreContainer: 
#with Columns[0]:
with st.sidebar:
	if st.button("Save"):
		if chain == True:
			st.warning('Unchain first!', icon="⚠️")
		else:
			#st.write(inputs)
			data_to_file = pd.DataFrame(inputs)
			data_to_file.to_csv("save.csv", index=True)
		#if st.button("Save II"):
		#	data_to_file = pd.DataFrame(inputs)
		#	data_to_file.to_csv("save2.csv", index=True)

#with SuperLeft:
with colGraph:
	tab1, Banco, BancoCap, PagaSolo = st.tabs(['Inv  ', '| Banco ', '| Banco+Cap ','| Paga solo ']) #PagaSolo
	with tab1:
		costos = [processed[n]['costo_prestamo'] for n in range(N)]
		xs_inv, ys_inv = proc.find_x_intercepts_for_y(mmatrix[0:N], targets=costos)

		names = [inputs[n]['name'] for n in NN]
		names.append("Opor")
		#names.append("Opor2")
		#names.append("Opor3")
		#names.append("Opor4")
		hides = [inputs[n]['hide_graph'] for n in NN]
		hides.extend(hides)
		#hides.append(False)
		#hides.append(False)

		disp.plot2(mmatrix, x_m, y_m, names=names, hides=hides, show_labels=True, labels=y_m)

	with Banco:
		disp.plot(inputs, smatrix, x_s, zeros, labels=x_s)
		st.write("Tiempo requerido para saldar cuentas con el banco.")

	with BancoCap:
		disp.plot_duo(inputs, duo_matrix, x_duo, y_duo, labels=x_duo)
		st.write("Tiempo para volver a tener el mismo capital inicial.  \nIncluye ahorros (desembolso) mensuales.")

	with PagaSolo:
		disp.plot(inputs, pmatrix, x_p, y_p, labels=x_p)
		st.write("-Costo de préstamo + valor y producción de la inversión.")