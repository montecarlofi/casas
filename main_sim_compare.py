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
MONEY_MULTIPLIER = 1000
N = 4

# compliance probability (investing the same amount in a non-house-buying situation)
# interés variable
# mc sim on: interest, ingresos, occupancy, stock market

# problem with (i+1) vs (i+i) in some series

cols = []
_ = [(cols.append(2), cols.append(1)) for x in range(N)]
cols.append(np.array(cols).sum())
#colA, colempty, colB, colempty2, colC, colempty3, colD, colempty4, colGraph = st.columns([2, 1, 2, 1, 2, 1, 2, 1, 8])
colA, colempty, colB, colempty2, colC, colempty3, colD, colempty4, colGraph = st.columns(cols)
##colA, colempty, colB, colempty2, colGraph = st.columns([2, 1, 2, 1, 6])
columns = [colA, colB, colC, colD]


inputs = get.read_data()[0:N]
readdata = get.read_data()[0:N]
colnames = [inputs[i]['name'] for i in range(N)]
ingresos = [MONEY_MULTIPLIER*inputs[i]['ingreso_pesimista'] for i in range(N)]
inversiones = [inputs[i]['inversion'] for i in range(N)]
tasas = [inputs[i]['tasa'] for i in range(N)]
#porciones = [inputs[i]['porcion'] for i in range(N)]
#meses_con_prestamos = [inputs[i]['meses_con_prestamo'] for i in range(N)]
retrasos = [inputs[i]['retraso'] for i in range(N)]
valorizaciones = [inputs[i]['valorizacion'] for i in range(N)]
hide_graphs = [inputs[i]['hide_graph'] for i in range(N)]
max_desembolso_mensual = inputs[0]['max_desembolso_mensual'] # max_desembolso_mensual = [inputs[n]['max_desembolso_mensual'] for n in range(N)]
try:
	cap_inis = [inputs[i]['cap_ini'] for i in range(N)]
except Exception as e:
	cap_inis = [0 for i in range(N)]

processed = [{} for i in range(N)]

# Names
for i in range(N):
	with columns[i]:
		key = "Sim" + str(i)
		colnames[i] = st.text_input('', value=colnames[i], key=key) # colnames[i] = st.text_input('', value=f"simname{i}", key=f"simname{i}")
		hide_graph = st.checkbox("", value=hide_graphs[i], key=f'hide_graph_{key}', on_change=None, disabled=False)
		inputs[i].update({ 'hide_graph': hide_graph })

with st.sidebar:
	st.subheader("SM Invs.  \n  ")

	expanders = [st.expander(label=f'Inv: {colnames[i]}') for i in range(N)]
	for i in range(N):
		with expanders[i]:
			inputs[i] = proc.input_sidebar(colnames[i], inversion=inversiones[i], cap_propio=cap_inis[i], tasa=tasas[i], hide_graph=inputs[i]['hide_graph'])#, max_desembolso_mensual=max_desembolso_mensual)

	st.markdown("""---""")

	max_desembolso_mensual = st.number_input('Max desembolso mensual', 0, 30, max_desembolso_mensual)
	_ = [inputs[n].update({ 'max_desembolso_mensual': max_desembolso_mensual}) for n in range(N)]
	meses_display = st.selectbox('Ilustración (meses)', np.arange(60, 721, 60), index=4, key="meses_display")

	st.markdown("""---""")
	r_mes_opor = st.slider("Crec. opor. alt.", 0, 15, 0, 1)
	r_mes_opor = 1 + (r_mes_opor/100)
	r_mes_opor = np.e**(np.log(r_mes_opor)/12)

columns = [colA, colB, colC, colD]
for i in range(N):
	with columns[i]:
		#st.subheader(f'_{colnames[i]}_')
		inputs[i].update(proc.input_cols(colnames[i], 
			hide_graph=inputs[i]['hide_graph'], 
			ingreso=ingresos[i], 
			retraso=retrasos[i],# retraso=disk_inputs[i]['retraso'], 
			valorizacion=valorizaciones[i],#)) # valorizacion=disk_inputs[i]['valorizacion']))
			max_desembolso_mensual=inputs[i]['max_desembolso_mensual']))


# Matrices setup
#mmatrix = np.zeros([N, MAX_LENGTH])
#inputs[0]['inversion'] = 1000
mmatrix = proc.sim_main(inputs, MAX_LENGTH)
imatrix = np.zeros([N, MAX_LENGTH])
sin_val_matrix = proc.sim_main(inputs, MAX_LENGTH)
#repay_matrix = np.zeros([N, MAX_LENGTH])
#pagar_solo_matrix = np.zeros([N, MAX_LENGTH])

# Find minimum repayment times.
# Each steps represents the total loan+cost if it were to be paid in x steps. (1, END+1)
loan_matrix = proc.loan_matrix(inputs, MAX_LENGTH) 
print(readdata[0])
#_ = [inputs[n].update({'max_desembolso_mensual': 10}) for n in range(N)]
xs_repay, ys_repay = proc.calc_min_repay_times(inputs, loan_matrix)

# Price of loans.
Cs = [inputs[n]['inversion'] - (inputs[n]['cap_ini'] + inputs[n]['max_desembolso_mensual']*inputs[n]['retraso']) for n in range(N)]
#print("Y REP ", ys_repay[0]); #exit()
# If loan/cap_ini is very high, then ys_repay will be outside MAX_LENGTH

_ = [processed[n].update({ 'costo_prestamo': ys_repay[n] - Cs[n] }) for n in range(N)]

# Deuda y capital
_ = [processed[n].update({'deuda_y_capital': inputs[n]['inversion'] + processed[n]['costo_prestamo']}) for n in range(N)]

# Matrix for how long it takes to recover cash and capital spent. No compounding included.
# Then make flipped (- to 0) matrix.
months_paid = [xs_repay[n] + inputs[n]['retraso'] for n in range(N)]
cash_spent = [months_paid[n]*inputs[n]['max_desembolso_mensual'] + inputs[n]['cap_ini'] for n in range(N)]
_ = [processed[n].update({ 'months_paid': months_paid[n] }) for n in range(N)]
_ = [processed[n].update({ 'cash_spent': cash_spent[n] }) for n in range(N)]
# Income only
income_only_matrix = proc.sim_income_only(inputs, MAX_LENGTH)
m = np.array(cash_spent).reshape(N, 1)
inverse_income_matrix = -m[::] + income_only_matrix[:]
xs_inverse_income, ys_inverse_income = proc.find_x_intercepts_for_y(inverse_income_matrix, targets=[0 for n in range(N)])
inverse_income_matrix = proc.cut_after(inverse_income_matrix, targets=[0 for n in range(N)])

# Income+desembolso
income_and_desembolso_matrix = proc.sim_income_and_desembolso(inputs, MAX_LENGTH)
m = np.array(cash_spent).reshape(N, 1)
income_and_desembolso_matrix_inverse = -m[::] + income_and_desembolso_matrix[:]
xs_inverse_all, ys_inverse_all = proc.find_x_intercepts_for_y(income_and_desembolso_matrix, targets=[0 for n in range(N)])
income_and_desembolso_matrix_inverse = proc.cut_after(income_and_desembolso_matrix_inverse, targets=[0 for n in range(N)])

#print("M ", months_paid[0])
#print("Cash spent ", cash_spent[0])
#print("Deuda y cap ", processed[2]['deuda_y_capital'])


# Reverse matrix DOES NOT WORK IN GRAPH! WHY?
inverse_mmatrix = -m[::] + mmatrix[:]
inverse_mmatrix = mmatrix.copy()
xs_inverse, ys_inverse = proc.find_x_intercepts_for_y(inverse_mmatrix, targets=[0 for n in range(N)])
inverse_mmatrix = proc.cut_after(inverse_mmatrix, targets=[0 for n in range(N)])

# Shift mmatrix below 0
# Start points are one step too early....................
retrasos = [inputs[n]['retraso'] for n in range(N)]
targets = [processed[n]['cash_spent'] + inputs[n]['inversion'] for n in range(N)]
xs_invm, ys_invm = proc.find_x_intercepts_for_y(mmatrix, targets=targets)
for key in ys_invm:
	ys_invm[key] = 0 - targets[key]
start_points = [mmatrix[n][0+retrasos[n]] for n in range(N)]
start_points = np.array(start_points).reshape(N, 1)
inverse_mmatrix = mmatrix[:] - 2*start_points
inverse_mmatrix = proc.cut_after(inverse_mmatrix, targets=[0 for n in range(N)])

# Matrix for how long it takes to recover cash and capital spent. No compounding included.
# Then make flipped (- to 0) matrix.
caps = np.array([inputs[n]['cap_ini'] for n in range(N)]).reshape(N, 1)
inverse_cap_matrix = -caps[::] + income_only_matrix[:]
xs_inverse_cap, ys_inverse_cap = proc.find_x_intercepts_for_y(inverse_cap_matrix, targets=[0 for n in range(N)])
inverse_cap_matrix = proc.cut_after(inverse_cap_matrix, targets=[0 for n in range(N)])

# Matrix for erasing debt.
debt_matrix = proc.sim_debt(inputs, xs_repay, ys_repay, MAX_LENGTH)
xs_debt = {}
for n in range(N):
	xs_debt.update({ n: xs_repay[n]+inputs[n]['retraso'] })
ys_debt = ys_repay.copy()
debt_matrix[:] = 0-debt_matrix

# Matrix for displaying how long it takes to recover initial capital spent.
incomes_only_shifted = proc.shift_sequences(income_only_matrix, shifts=xs_repay)
#print("Shifted Yes, ", incomes_only_shifted[2])
incomes_only_shifted_inverse = -caps[:] + incomes_only_shifted[:]
targets = [0 for n in range(N)]
xs_recovercap, ys_recovercap = proc.find_x_intercepts_for_y(incomes_only_shifted_inverse, targets=targets)
#print("Reocver intercepts ", xs_recovercap)
incomes_only_shifted_inverse = proc.cut_after(incomes_only_shifted_inverse, targets=targets)


# 117, not 120. See: print(incomes_only_shifted_inverse[2]); exit()

# Matrix for displaying how long it takes to recover initial capital spent, 
# incorporating monthly savings of 'max_desmobolso_mensual'.


#Llegar a 0 incl CAP

#just one matrix
#copy, find targets

#volver a cap_ini in lighter colour


##for i in range(mmatrix.shape[0]): # For each row
	#mmatrix[i] = sim00(inversiones[i], ingresos[i], rates_mes[i], MAX_LENGTH, retraso=retrasos[i])
##	mmatrix[i] = proc.sim00(inputs[i]['inversion'], inputs[i]['ingreso_pesimista'], inputs[i]['r_mes'], MAX_LENGTH, retraso=inputs[i]['retraso'])

# Find months (x_intercepts) to pay off loan at max_desembolso_mensual
amorts, x_intercepts, y_intercepts = proc.intercepts_llegar_a_0(inputs, MAX_LENGTH) # {dicts}

for i in range(N):
	imatrix[i] = proc.sim_llegar_a_0(inputs[i], x_intercepts[i], y_intercepts[i], MAX_LENGTH)


costos_prestamos = proc.costos_prestamos(inputs, x_intercepts, y_intercepts)
_ = [processed[n].update({'costo_prestamo': costos_prestamos[n]}) for n in range(N)]
#processed[0].update({'costo_prestamo': costos_prestamos[0]})


desembolsos = [inputs[i]['max_desembolso_mensual'] for i in range(N)]
#x_intercepts, y_intercepts
#123
###############print(x_intercepts)

###[processed[n].update({'amort': amorts[n]}) for n in range(N)]
#[processed[n].update({'deuda_y_capital': amorts[n]*x_intercepts[n]+inputs[n]['cap_ini']}) for n in range(N)]


retrasos = [inputs[x]['retraso'] for x in range(N)]
x_intercepts0, y_intercepts0 = proc.intercepts(mmatrix, targets=[processed[x]['costo_prestamo'] for x in range(N)], retrasos=retrasos)
x_interceptspagasolo, y_interceptspagasolo = proc.intercepts(mmatrix, targets=[processed[x]['deuda_y_capital'] for x in range(N)], retrasos=retrasos)
del retrasos


deudYcap = [processed[n]['deuda_y_capital'] for n in range(N)]
deudYcap = np.array(deudYcap)
deudYcap = deudYcap.reshape((N, 1))
pagar_solo_matrix = mmatrix[:] - deudYcap ####################### WAS HERE




# Costos de oportunidades
###derrames = [inputs[i]['ingreso_pesimista'] - processed[i]['amort'] for i in range(N)]
###for i in range(len(derrames)):
###	if derrames[i] >= 0:
###		derrames[i] = 0.000000001
###derrames_totales = [derrames[i]*x_intercepts[i] for i in range(N)]
#, derrames[1]*inputs[0]['meses_con_prestamo'], derrames[2]*inputs[1]['meses_con_prestamo'], derrames[3]*inputs[1]['meses_con_prestamo']]



# Oportunidad # ##########################3······ FIX !
opors = [0, 0, 0, 0]
###costos_de_oportunidad = [-inputs[n]['inversion'] + derrames_totales[n] for n in range(N)]
capitales = [inputs[n]['inversion'] for n in range(N)]
#costos_de_oportunidad = [-inv*(1-portion) + derrame_total for inv, portion, derrame_total in zip([inputs[0]['inversion'], inputs[0]['inversion'], inputs[1]['inversion'], inputs[1]['inversion']], [inputs[0]['porcion'], inputs[0]['porcion'], inputs[1]['porcion'], inputs[1]['porcion']], derrames_totales)]
#geo = lambda P, r, n: P * ((1-r**n) / (1-r))
geo = lambda P, r, n: P * (1-r**n) / (1-r)
#inputs[0]['r_mes']_opor = st.slider("Crecimiento A", 0, 15, 0, 1)
#r_mes_B_opor = st.slider("Crecimiento B", 0, 15, 0, 1)
mesesconprest = x_intercepts.copy()

if r_mes_opor == 0:
	opors = [capitales[n] + mesesconprest[n] * -derrames[n] for n in range(N)]
else:
	r = 1 + (r_mes_opor/100)
	r = np.e**(np.log(r_mes_opor)/12)
	####opors = [geo(-derrames[n], r, x_intercepts[n]) + capitales[n]*r**x_intercepts[n] for n in range(N)]

#if r_mes_A_opor == 0:
#	opors[0] = capitales[0] + inputs[0]['meses_con_prestamo'] * -derrames[0]
#	opors[1] = capitales[1] + inputs[0]['meses_con_prestamo'] * -derrames[1]
#else:
#	r_mes_A_opor = 1 + (r_mes_A_opor/100)
#	r_mes_A_opor = np.e**(np.log(r_mes_A_opor)/12)
#	opors[0] = geo(-derrames[0], r_mes_A_opor, x_intercepts[0]) + capitales[0]*r_mes_A_opor**x_intercepts[0]
#	opors[1] = geo(-derrames[1], r_mes_A_opor, x_intercepts[0]) + capitales[1]*r_mes_A_opor**inputs[0]['meses_con_prestamo']
#if r_mes_B_opor == 0:
#	opors[2] = capitales[2] + inputs[1]['meses_con_prestamo'] * -derrames[2]
#	opors[3] = capitales[3] + inputs[1]['meses_con_prestamo'] * -derrames[3]
#else:
#	r_mes_B_opor = 1 + (r_mes_B_opor/100)
#	r_mes_B_opor = np.e**(np.log(r_mes_B_opor)/12)
#	opors[2] = geo(-derrames[2], r_mes_A_opor, inputs[1]['meses_con_prestamo']) + capitales[2]*r_mes_B_opor**inputs[1]['meses_con_prestamo']
#	opors[3] = geo(-derrames[3], r_mes_A_opor, inputs[1]['meses_con_prestamo']) + capitales[3]*r_mes_B_opor**inputs[1]['meses_con_prestamo']





# Opportunity-cost matrix
#cost_opor_matrix = np.zeros([N, MAX_LENGTH]) # cost_opor_matrix = np.zeros([4, length])
#tapacuotas = np.array([inputs[i]['derrame_hasta_pagar_prestamo'] for i in range(N)])
#tapacuotas = tapacuotas.reshape((N, 1))

#capitales = [inputs[i]['capital'] for i in range(N)]
#caps = np.array(capitales)
#caps = caps.reshape((N, 1))
#rs = np.array([r_mes_opor for _ in range(N)])
#for i in range(len(rs)):
#	if rs[i] == 0:
#		rs[i] = 1
#rs = rs.reshape((N, 1))

#cost_opor_matrix[0:N,0:1] = caps # First column.
#for i in range(1, length):
#	cost_opor_matrix[0:N,i:i+1] = cost_opor_matrix[0:N,i-1:i]*rs + tapacuotas[0:N,:]

#for i in range(length, MAX_LENGTH):
#	cost_opor_matrix[0:N,i:i+1] = cost_opor_matrix[0:N,i-1:i]*rs



# Crop matrix
mmatrix = mmatrix[0:N,0:meses_display+1]
#inverse_mmatrix[:] = -2*inverse_mmatrix[:]
#inverse_mmatrix[:] = inverse_mmatrix[:]-inverse_mmatrix[:]-inverse_mmatrix[:]

inverse_mmatrix = inverse_mmatrix[0:N,0:proc.len_longest_graph(inverse_mmatrix)+11]


#inverse_mmatrix = inverse_mmatrix[0:N,0:proc.len_longest_graph(inverse_mmatrix)+11]
imatrix = imatrix[0:N,0:proc.len_longest_graph(imatrix)+11] #imatrix = proc.cut_after_longest(imatrix, extra=10)
inverse_cap_matrix = inverse_cap_matrix[0:N,0:proc.len_longest_graph(inverse_cap_matrix)+11]
inverse_income_matrix = inverse_income_matrix[0:N,0:proc.len_longest_graph(inverse_income_matrix)+11]
incomes_only_shifted_inverse = incomes_only_shifted_inverse[0:N,0:proc.len_longest_graph(incomes_only_shifted_inverse)+11]
debt_matrix = debt_matrix[0:N,0:proc.len_longest_graph(debt_matrix)+11]
income_and_desembolso_matrix_inverse = income_and_desembolso_matrix_inverse[0:N,0:proc.len_longest_graph(income_and_desembolso_matrix_inverse)+11]


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


columns = [colA, colB, colC, colD]
for i in range(len(columns)):
	with columns[i]:
		m = meses_display-1
		m = meses_display
		#crecimiento_anual_alt = np.e**(np.log(mmatrix[i][m]/(inputs[i]['deuda_y_capital']))/(m/12))
		#crecimiento_anual_alt = np.e**(np.log(mmatrix[i][m]/(processed[i]['deuda_y_capital']))/(m/12))
		crecimiento_anual_alt = np.e**(np.log(mmatrix[:,-1][i]/(processed[i]['cash_spent']))/(m/12))
		st.write(round(crecimiento_anual_alt, 4)) #st.write(round(mmatrix[0][m]), "M")
		st.write(xs_repay[i], " meses")
		#st.write(round(-processed[i]['deuda_y_capital']))

# Save/restore
with colA:
	if st.button("Save"):
		data_to_file = pd.DataFrame(inputs)
		data_to_file.to_csv("save.csv", index=True)
	#if st.button("Save2"):
	#	data_to_file = pd.DataFrame(inputs)
	#	data_to_file.to_csv("save2.csv", index=True)

with colGraph:
	tab1, tab2, tab3, tab4 = st.tabs(['Inv  ', '| PagarBanco ', '| Rec: Cap ', '| Rec: Todo (!+crecmto!) ']) #PagaSolo
	with tab1:
		costos = [processed[n]['costo_prestamo'] for n in range(N)]
		xs_inv, ys_inv = proc.find_x_intercepts_for_y(mmatrix, targets=costos)

		#disp.plot(inputs, mmatrix, x_intercepts0, y_intercepts0)
		disp.plot(inputs, mmatrix, {}, {})

	with tab2:
		#disp.plot_llegar_a_0(inputs, x_intercepts, imatrix, N)
		#disp.recuperar_capital(inputs, x_intercepts, imatrix, N)
		disp.plot(inputs, debt_matrix, xs_debt, ys_debt)
		#_ = [print(key, ys_debt[key]) for key in ys_debt]

	with tab3:
		#disp.plot(inputs, inverse_cap_matrix, xs_inverse_cap, ys_inverse_cap)
		
		#disp.plot(inputs, income_and_desembolso_matrix_inverse, xs_inverse_all, ys_inverse_all)
		disp.plot(inputs, incomes_only_shifted_inverse, xs_recovercap, ys_recovercap)
		st.write("Incluye ahorros (desembolso) mensuales.")

	with tab4:
		# No valuation
		disp.plot(inputs, inverse_income_matrix, xs_inverse_income, ys_inverse_income)
		disp.plot(inputs, inverse_mmatrix, xs_invm, ys_invm)

		# Yes valuation
		#disp.plot(inputs, inverse_mmatrix, xs_inverse, ys_inverse)		

		#disp.plot_llegar_a_0(inputs, x_interceptspagasolo, mmatrix, N)

	# Graph debt remaining
