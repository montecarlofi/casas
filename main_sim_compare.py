#import display.line_graph as line_graph
import streamlit as st; st.set_page_config(layout="wide")
import altair as alt
#import display.hide_streamlit #as st_hide
import streamlit as st; #st.set_page_config(layout="wide")
from streamlit_option_menu import option_menu
import numpy as np 
import plotly.express as px
import pandas as pd; #import get_data
import matplotlib.pyplot as plt
from numpy.random import default_rng
RNG = default_rng().standard_normal

MAX_LENGTH = 1200

# costo de oportunidad ############################# 
# costo de op: ROI
# compliance probability (investing the same amount in a non-house-buying situation)
# mc sim on: interest, ingresos, occupancy, stock market

#colA, colB, colD, colGraph = st.columns([1, 1, 1, 5], gap="large") # gap does not work with older Python versions.
colA, colempty, colB, colempty2, colGraph = st.columns([2, 1, 2, 1, 6])

with st.sidebar:
	expanderA = st.expander(label='Inversión A')
	with expanderA:
		invA = st.number_input('Monto A', value=400)
		interes_banco_A = st.number_input('Tasa bancaria A', value=18.70) / 100
		portionA = st.slider('% prestado A', value=70) / 100
		P_A = invA * portionA
		anos_con_prestamo_A = st.slider("Años con préstamo A", 1, 30, 4, 1)
		meses_con_prestamo_A = anos_con_prestamo_A * 12 

	expanderB = st.expander(label='Inversión B')
	with expanderB:
		invB = st.number_input('Inversión B', value=380)
		interes_banco_B = st.number_input('Tasa bancaria B', value=18.7) / 100
		portionB = st.slider('% prestado', value=70) / 100
		P_B = invB * portionB
		anos_con_prestamo_B = st.slider("Años con préstamo ", 1, 30, 3, 1, key="anos_B")
		meses_con_prestamo_B = anos_con_prestamo_B * 12 	

	st.markdown("""---""")
	meses_display = st.selectbox('Ilustración (meses)', np.arange(60, 721, 60), index=1, key="meses_display")

with colA:
	st.subheader('_Sim A_')
	#st.write("───── Inversión A ─────")
	valorizacionA = st.slider("Valorización A", -5, 15, 0, 1)
	valorizacionA = 1 + (valorizacionA/100)
	r_mes_A = np.e**(np.log(valorizacionA)/12)
	#rA = r_mes_A - 1

	ingreso_pesimista_A = st.slider("Ingreso pesimista K", 0, 5000, 2400, 100)
	ingreso_optimista_A = st.slider("Ingreso optimista K", 0, 6000, 3500, 100)
	ingreso_pesimista_A /= 1000
	ingreso_optimista_A /= 1000

	retrasos = []
	retrasoA = st.slider("Retraso en meses A", 0, 36, 0, 6)
	retrasos.append(retrasoA); retrasos.append(retrasoA)


	#st.markdown("""---""")
	#r_mes_A_opor = st.slider("Crecimiento A", 0, 15, 0, 1)

with colB:
	st.subheader('_Sim B_')
	#st.write("───── Inversión B ─────")
	valorizacionB = st.slider("Valorización B", -5, 15, 0, 1, key='valorizacionB')#, label_visibility="visible")
	valorizacionB = 1 + (valorizacionB/100)
	r_mes_B = np.e**(np.log(valorizacionB)/12)
	#rB = r_mes_B - 1

	#loan_interest = st.slider("Interés anual bancario ", 1, 20, 8, 1) / 100
	#loan_interest = st.selectbox(label, [x for x in range(10)], index=0, format_func=special_internal_function, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible")
	#loan_interest = st.selectbox('Tasa bancaria', np.arange(1, 20, 0.25), index=27)

	ingreso_pesimista_B = st.slider("Ingreso pesimista K", 0, 5000, 2800, 100, key='ingreso_pesimista_B')
	ingreso_optimista_B = st.slider("Ingreso optimista K", 0, 5000, 4000, 100, key='ingreso_optimista_B')
	ingreso_pesimista_B /= 1000
	ingreso_optimista_B /= 1000

	retrasoB = st.slider("Retraso en meses B", 0, 36, 13, 6)
	retrasos.append(retrasoB); retrasos.append(retrasoB)

	#st.markdown("""---""")
	#r_mes_B_opor = st.slider("Crecimiento B", 0, 15, 0, 1)

	#linknames = ['A', 'B', 'C']
	#the_icons = ['globe', 'gear', 'geo-fill'] # ['globe'] + ['geo-fill' for x in range(len(runners))] + ['gear']
	#selected2 = option_menu(None, linknames, 
	#        icons=the_icons, menu_icon="cast", default_index=1)
	#selected2

rates_mes = [r_mes_A, r_mes_A, r_mes_B, r_mes_B]

ingresos = [ingreso_pesimista_A, ingreso_optimista_A, ingreso_pesimista_B, ingreso_optimista_B]

inversiones = [invA, invA, invB, invB]
porciones = [portionA, portionA, portionB, portionB]

# Capitales
#capitales = [inv*(1-portion) for inv, portion in zip([invA, invA, invB, invB], [portionA, portionA, portionB, portionB])]
capitales = [inv*(1-portion) for inv, portion in zip(inversiones, porciones)]

prestamos = [(inv * porcion) for inv, porcion in zip(inversiones, porciones)]

# Amortización
def amort_mes(i, n, P):
	i /= 12
	return (P*i*(1+i)**n) / ((1+i)**n - 1)

amort_mes_A   = amort_mes(interes_banco_A, meses_con_prestamo_A, P_A)
amort_mes_B   = amort_mes(interes_banco_B, meses_con_prestamo_B, P_B)
amort_total_A = amort_mes_A * meses_con_prestamo_A
amort_total_B = amort_mes_B * meses_con_prestamo_B
amorts = [amort_total_A, amort_total_A, amort_total_B, amort_total_B]

outlay_A = invA * (1-portionA) + amort_total_A
outlay_B = invB * (1-portionB) + amort_total_B
outlay_A = amort_total_A
outlay_B = amort_total_B
deuda_A = amort_total_A
deuda_B = amort_total_B

deudas = amorts.copy()

deudas_y_capitales = [a+b for a, b in zip(deudas, capitales)]

#intereses_totales = [am-cap for am, cap in zip(amorts, capitales)]
intereses_totales = [am-pres for am, pres in zip(amorts, prestamos)]

# Costos de oportunidades
derrames = [ingreso_pesimista_A - amort_mes_A, ingreso_optimista_A - amort_mes_A, ingreso_pesimista_B - amort_mes_B, ingreso_optimista_B - amort_mes_B]
for i in range(len(derrames)):
	if derrames[i] >= 0:
		derrames[i] = 0.000000001
derrames_totales = [derrames[0]*meses_con_prestamo_A, derrames[1]*meses_con_prestamo_A, derrames[2]*meses_con_prestamo_B, derrames[3]*meses_con_prestamo_B]

# Costo de préstamos
costo_A = deuda_A - invA*portionA
costo_B = deuda_B - invB*portionB
costos_prestamos = [costo_A, costo_A, costo_B, costo_B]

with colA:
	#message  = f'{labels[0]}: {derrames[0]:.2f}/mes ({derrames_totales[0]:.0f})'
	#message  = f'A1: {derrames[0]:.2f}/mes ({derrames_totales[0]:.0f})'
	#st.write(message)
	#st.write(-deuda_A-capitales[0])

	#st.metric(label="De bolsillo mes (tot):", value="", delta=message, delta_color="off")#, delta_color="inverse")

	st.markdown("""---""")
	r_mes_A_opor = st.slider("Crecimiento A", 0, 15, 0, 1)

with colB:
	#message2 = f'B1: {derrames[2]:.2f}/mes ({derrames_totales[2]:.0f})'
	#st.write(message2)
	#st.write(-deuda_B-capitales[2])

	st.markdown("""---""")
	r_mes_B_opor = st.slider("Crecimiento B", 0, 15, 0, 1)

# Oportunidad
opors = [0, 0, 0, 0]
costos_de_oportunidad = [-inv*(1-portion) + derrame_total for inv, portion, derrame_total in zip([invA, invA, invB, invB], [portionA, portionA, portionB, portionB], derrames_totales)]
#geo = lambda P, r, n: P * ((1-r**n) / (1-r))
geo = lambda P, r, n: P * (1-r**n) / (1-r)
#r_mes_A_opor = st.slider("Crecimiento A", 0, 15, 0, 1)
#r_mes_B_opor = st.slider("Crecimiento B", 0, 15, 0, 1)

if r_mes_A_opor == 0:
	opors[0] = capitales[0] + meses_con_prestamo_A * -derrames[0]
	opors[1] = capitales[1] + meses_con_prestamo_A * -derrames[1]
else:
	r_mes_A_opor = 1 + (r_mes_A_opor/100)
	r_mes_A_opor = np.e**(np.log(r_mes_A_opor)/12)
	opors[0] = geo(-derrames[0], r_mes_A_opor, meses_con_prestamo_A) + capitales[0]*r_mes_A_opor**meses_con_prestamo_A
	opors[1] = geo(-derrames[1], r_mes_A_opor, meses_con_prestamo_A) + capitales[1]*r_mes_A_opor**meses_con_prestamo_A
if r_mes_B_opor == 0:
	opors[2] = capitales[2] + meses_con_prestamo_B * -derrames[2]
	opors[3] = capitales[3] + meses_con_prestamo_B * -derrames[3]
else:
	r_mes_B_opor = 1 + (r_mes_B_opor/100)
	r_mes_B_opor = np.e**(np.log(r_mes_B_opor)/12)
	opors[2] = geo(-derrames[2], r_mes_A_opor, meses_con_prestamo_B) + capitales[2]*r_mes_B_opor**meses_con_prestamo_B
	opors[3] = geo(-derrames[3], r_mes_A_opor, meses_con_prestamo_B) + capitales[3]*r_mes_B_opor**meses_con_prestamo_B


	#print(opors); exit()
	#print(costos_de_oportunidad); exit()
	#costo_oportunidad_A1 = -invA * (1-portionA) + derrames_totales[0]


#print(derrames)
#print(amort_total_A, amort_total_B)
#print(outlay_A, outlay_B)#; exit()

# Matrices setup
mmatrix = np.zeros([4, MAX_LENGTH])
#print(outlay_A, outlay_B)
#print(costos_prestamos[0]); exit()
def sim(inv, deuda, ingreso, r, n, meses_con_prestamo):
	y = np.zeros([1, n])
	#y[0][i] = -deuda
	#for i in range(1, n):
	for i in range(0, n):
		y[0][i] = -deuda + ingreso*i + inv*r**i - inv
	return y

def sim0(inv, cap, costo_prestamo, ingreso, r, n, meses_con_prestamo):
	y = np.zeros([1, n])
	for i in range(0, n):
		#y[0][i] = -costo_prestamo + ingreso*i + inv*r**i #- inv#- cap
		y[0][i] = ingreso*i + inv*r**i #- inv#- cap
	return y

def sim00(inv, ingreso, r, n, retraso=0):
	y = np.zeros([1, n])
	if retraso == 0:
		start = 0
	else:
		start = retraso
		for i in range(0, start):
			y[0][i] = None
	for i in range(start, n):
		y[0][i] = ingreso*(i-start) + inv*r**(i-start)
	return y

def sim_paga_solo(inv, debt_and_cap, ingreso, r, n, retraso=0):
	#print(f"Debt and cap is {debt_and_cap}")
	y = np.zeros([1, n])

	if retraso == 0:
		start = 0
	else:
		start = retraso
		for i in range(0, start):
			y[0][i] = None

	for i in range(start, n): # Could go out of bounds because of i+1 and i+2 ?
		y[0][i] = -debt_and_cap + ingreso*(i-start) + inv*r**(i-start) - inv
		#print(y[0][i])
		if y[0][i] >= 0:#debt_and_cap:
			#y = y[0:1,0:i+1]
			y[0][i+1] = -debt_and_cap + ingreso*(i-start) + inv*r**(i-start) - inv
			y[0:1,0:i+2] = y[0:1,0:i+2]
			y[0:1,i+1:] = None

			#y[0:1,0:i+1] = y[0:1,0:i+1]
			#y[0:1,i:] = None
			break
	return y

def sim_step(capital, deuda, meses_con_prestamo, ingreso, r, n):
	y = np.zeros([1, n])
	# y = np.zeros([2, n]) # Line 0: MonteCarlo growth on capital. Line 1: Total (MC + ingresos)
	y[0][0] = -deuda
	for i in range(1, n):
		#y[0][i] = y[0][i-1] + y[0][i-1]* + ingreso 
		y[0][i] = y[0][i-1] + y[0][i-1]*r - capital + ingreso 
	return y

def sim_step0(inv, cost, ingreso, r, n):
	y = np.zeros([1, n]) # y = np.zeros([2, n]) # Line 0: MonteCarlo growth on capital. Line 1: Total (MC + ingresos)
	y[0][0] = inv #- cost
	for i in range(1, n):
		y[0][i] = y[0][i-1] + y[0][i-1]*(r-1) + ingreso 
	return y

for i in range(mmatrix.shape[0]): # For each row
	mmatrix[i] = sim00(inversiones[i], ingresos[i], rates_mes[i], MAX_LENGTH, retraso=retrasos[i])

imatrix = mmatrix.copy()
for i in range(imatrix.shape[0]): # For each row
	imatrix[i] = sim_paga_solo(inversiones[i], deudas_y_capitales[i], ingresos[i], rates_mes[i], MAX_LENGTH, retraso=retrasos[i])


# Get intercepts
#imatrix = np.zeros([4, MAX_LENGTH])

def get_intercepts(outlays, imatrix): # Problem: This includes valuation.
	xs = {}
	ys = {}
	n = imatrix.shape[1]
	for k in range(4):
		i = 0
		for i in range(0, n+0):
			if imatrix[k][i] >= 0:
			#if imatrix[k][i] - outlays[k] >= 0:
				xs[k] = i
				ys[k] = imatrix[k][i]
				break
	return xs, ys 

def get_intercepts2(outlays, imatrix): # Problem: This includes valuation.
	xs = {}
	ys = {}
	n = imatrix.shape[1]
	for k in range(4):
		i = 0
		for i in range(0, n+0):
			#if imatrix[k][i] >= 0:
			#if imatrix[k][i] - outlays[k] >= 0:
			if imatrix[k][i] >= outlays[k]:
				xs[k] = i
				ys[k] = imatrix[k][i]
				break
	return xs, ys 

def intercepts(mtrx, targets, retrasos=retrasos):
	xs = {}
	ys = {}
	n = mtrx.shape[1]
	m = mtrx.shape[0]

	for k in range(m):
		if retrasos[k] == 0:
			start = 0
		else:
			start = retrasos[k]
		for i in range(0, n+0-start):
			#print(mtrx[k][i])
			if mtrx[k][i+start] - mtrx[k][0+start] >= targets[k]: # Current - start | mtrx[k][i] - targets[k] <= 0
				#print("it's bigger")
				xs[k] = i+start
				ys[k] = mtrx[k][i+start]
				break
	#print(xs); # 350 - 401
	return xs, ys 

#print(deudas)
#st.write("Deudas y capitales", deudas_y_capitales)
#imatrix[0] = sim(invA, outlay_A, ingreso_pesimista_A, 1, amort_mes_A, MAX_LENGTH, meses_con_prestamo_A)
#imatrix[1] = sim(invA, outlay_A, ingreso_optimista_A, 1, amort_mes_A, MAX_LENGTH, meses_con_prestamo_A)
#imatrix[2] = sim(invB, outlay_B, ingreso_pesimista_B, 1, amort_mes_B, MAX_LENGTH, meses_con_prestamo_B)
#imatrix[3] = sim(invB, outlay_B, ingreso_optimista_B, 1, amort_mes_B, MAX_LENGTH, meses_con_prestamo_B)
#print("InvA + costo: ", invA+costos_prestamos[0])
x_intercepts, y_intercepts = get_intercepts([outlay_A, outlay_A, outlay_B, outlay_B], imatrix)
x_intercepts, y_intercepts = intercepts(imatrix, targets=deudas_y_capitales, retrasos=retrasos) # deudas_y_capitales
x_intercepts0, y_intercepts0 = intercepts(mmatrix, targets=costos_prestamos, retrasos=retrasos)
#length = imatrix[0].shape[0]
#print("Length ", length); exit()
#for i in range(imatrix.shape[0])
#	x_intercepts = imatrix[i].shape[0]
#	y_intercepts = imatrix[i][-1]


income_x_intercepts, income_y_intercepts = get_intercepts([outlay_A, outlay_A, outlay_B, outlay_B], mmatrix) # imatrix
#print(income_x_intercepts); 
#print(x_intercepts, y_intercepts)
#exit()

# Debt remaining
debt_matrix_A = np.zeros([2, meses_con_prestamo_A])
debt_matrix_B = np.zeros([2, meses_con_prestamo_B])
debt_matrix_A[0] = np.array([amort_total_A - -derrames[0]*x - ingreso_pesimista_A*x for x in range(meses_con_prestamo_A)])
debt_matrix_A[1] = np.array([amort_total_A - -derrames[1]*x - ingreso_optimista_A*x for x in range(meses_con_prestamo_A)])
debt_matrix_B[0] = np.array([amort_total_B - -derrames[2]*x - ingreso_pesimista_B*x for x in range(meses_con_prestamo_B)])
debt_matrix_B[1] = np.array([amort_total_B - -derrames[3]*x - ingreso_optimista_B*x for x in range(meses_con_prestamo_B)])
#print(debt_matrix_A)

# Debt matrix
length = meses_con_prestamo_A if meses_con_prestamo_A>meses_con_prestamo_B else meses_con_prestamo_B
debt_matrix = np.zeros([4, length])
debts = np.array([deuda_A, deuda_A, deuda_B, deuda_B])
debts = debts.reshape((4, 1))
amorts = np.array([amort_mes_A, amort_mes_A, amort_mes_B, amort_mes_B])
amorts = amorts.reshape((4, 1))
#print(amorts[0:4,:]); exit()

debt_matrix[0:4,0:1] = debts # First column.
for i in range(1, length):
	debt_matrix[0:4,i:i+1] = debt_matrix[0:4,i-1:i] - amorts[0:4,:]
#print(debt_matrix)

# Opportunity-cost matrix
cost_opor_matrix = np.zeros([4, MAX_LENGTH]) # cost_opor_matrix = np.zeros([4, length])
tapacuotas = np.array(derrames)
tapacuotas = tapacuotas.reshape((4, 1))
#capitals = np.array([invA, invA, invB,  invB])
#capitals = np.array([-inv*(1-portion) for inv, portion in zip([invA, invA, invB, invB], [portionA, portionA, portionB, portionB])])
caps = -np.array(capitales)
caps = caps.reshape((4, 1))
rs = np.array([r_mes_A_opor, r_mes_A_opor, r_mes_B_opor, r_mes_B_opor])
for i in range(len(rs)):
	if rs[i] == 0:
		rs[i] = 1
rs = rs.reshape((4, 1))
#rs[:] 
#print(rs)
#st.write("tapacuotas: ", tapacuotas)
cost_opor_matrix[0:4,0:1] = -caps # First column.
for i in range(1, length):
	cost_opor_matrix[0:4,i:i+1] = cost_opor_matrix[0:4,i-1:i]*rs + -tapacuotas[0:4,:]
for i in range(length, MAX_LENGTH):
	cost_opor_matrix[0:4,i:i+1] = cost_opor_matrix[0:4,i-1:i]*rs
#print(cost_opor_matrix); exit()


# Crop matrix
mmatrix = mmatrix[0:4,0:meses_display+1]
cost_opor_matrix = cost_opor_matrix[0:4,0:meses_display+1]
imatrix = imatrix[0:4,0:meses_display+1]
#dmatrix = dmatrix[0:4,0:meses_display+1]
#print(cost_opor_matrix)


# Plotly/Streamlit display graphs
data = []
labels = ['A_pes', 'A_opt', 'B_pes', 'B_opt', 'deuda_A', 'deuda_A', 'deuda_B', 'deuda_B', 'op_A1', 'op_A2', 'op_B1', 'op_B2']

# Investment graphs
for l in range(4):
	line_data = mmatrix[l]
	for i in range(line_data.shape[0]):
		d = { 'name': labels[l], 'mes': i, 'y': line_data[i], 'blob': 1 }
		data.append(d)

# Debt graphs
for l in range(0,4,2):
	line_data = debt_matrix[l]
	for i in range(line_data.shape[0]):
		d = { 'name': labels[l+4], 'mes': i, 'y': line_data[i], 'blob': 1 }
########data.append(d)

# Opportunity-cost graphs
for l in range(4):
	line_data = cost_opor_matrix[l]
	for i in range(line_data.shape[0]):
		d = { 'name': labels[l+8], 'mes': i, 'y': line_data[i] }
		data.append(d)

#print(data)

df = pd.DataFrame(data)#; print(df); exit()

import plotly.express as px
plot = px.line(df, x=df.mes, y=df.y, hover_name=df.name, color='name', #title='Simulador de inversiones', 
	color_discrete_map={ 'A_pes': 'blue', 'A_opt': 'blue', 'B_pes': 'red', 'B_opt': 'red', 'deuda_A': 'grey', 'deuda_B': 'grey', 'op_B1': 'pink', 'op_B2': 'pink', 'op_A1': 'lightblue', 'op_A2': 'lightblue'}
	)

#for i in range(len(x_intercepts)):
for key in x_intercepts0:
	#print(key, y_intercepts0[key])
	#if MAX_LENGTH >= x_intercepts0[i]:
	if key == 0 or key == 1:
		yshift=8
	else:
		yshift=8
	if x_intercepts0[key] <= mmatrix.shape[1]:
		plot.add_annotation(x=x_intercepts0[key], y=y_intercepts0[key],
			text=(f'{x_intercepts0[key]}'),
			showarrow=False,
			yshift=yshift)

plot.update_layout(showlegend=True)



# Plot llegar a 0
data = []
for l in range(4):
	line_data = imatrix[l]
	for i in range(line_data.shape[0]):
		d = { 'name': labels[l], 'mes': i, 'y': line_data[i], 'blob': 1 }
		data.append(d)

df = pd.DataFrame(data)#; print(df); exit()

plot0 = px.line(df, x=df.mes, y=df.y, hover_name=df.name, color='name',  
	color_discrete_map={ 'A_pes': 'blue', 'A_opt': 'blue', 'B_pes': 'red', 'B_opt': 'red', 'deuda_A': 'grey', 'deuda_B': 'grey', 'op_B1': 'pink', 'op_B2': 'pink', 'op_A1': 'lightblue', 'op_A2': 'lightblue'}
	)

#for i in range(len(x_intercepts)):
for key in x_intercepts:
	#print(key, y_intercepts[key])
	#if MAX_LENGTH >= x_intercepts[i]:
	if key == 0 or key == 1:
		yshift=8
	else:
		yshift=8
	if x_intercepts[key] <= imatrix.shape[1]:
		plot0.add_annotation(x=x_intercepts[key], y=y_intercepts[key],
			text=(f'{x_intercepts[key]}'),
			showarrow=False,
			yshift=yshift)

plot0.update_layout(showlegend=True)

# Oportunidad-matrix
#dfo = pd.DataFrame(cost_opor_matrix)
#cost_opor_matrix = cost_opor_matrix.reshape((cost_opor_matrix.shape[1], cost_opor_matrix.shape[0]))
#oportunity_plot = px.line(cost_opor_matrix)
#oportunity_plot = px.line(dfo, x=df.mes, y=dfo.y, hover_name=df.name, color='name', title='Simulador de inversiones', 
#	color_discrete_map={ 'A_pes': 'blue', 'A_opt': 'blue', 'B_pes': 'red', 'B_opt': 'red', 'deuda_A': 'grey', 'deuda_B': 'grey'}
#	)


with colA:
	m = meses_display-1
	creimiento_anual = np.e**(np.log(mmatrix[0][m]/(deuda_A+capitales[0]))/(m/12))
	st.write(round(mmatrix[0][m]), "M")
	st.write(creimiento_anual)

with colB:
	m = meses_display
	creimiento_anual = np.e**(np.log(mmatrix[2][m]/(deuda_B+capitales[2]))/(m/12))
	st.write(round(mmatrix[2][m]), "M")
	st.write(creimiento_anual)

with colGraph:
	tab1, tab2 = st.tabs(['Inv  ', '| PagaSolo '])
	with tab1:
		st.plotly_chart(plot, use_container_width=True)

		de_bolsillo = []
		for i in range(len(derrames)):
			de_bolsillo.append({labels[i]: derrames[i]})
		#de_bolsillo.append({'A'})

		df = pd.DataFrame(de_bolsillo)

		#st.table(pd.DataFrame(de_bolsillo))

		message  = f'A1: {derrames[0]:.2f}/mes ({derrames_totales[0]:.0f})'
		st.write(message)
		st.write(-deuda_A-capitales[0], color="red")
		message2 = f'B1: {derrames[2]:.2f}/mes ({derrames_totales[2]:.0f})'
		st.write(message2)
		st.write(-deuda_B-capitales[2])

	with tab2:
		st.plotly_chart(plot0, use_container_width=True)
	#st.plotly_chart(oportunity_plot)

	# Cuotas tapadas por nosotros
	#derrames_totales = [derrames[i]*income_x_intercepts[i] for i in range(len(income_x_intercepts))]
	#derrames_totales = [d * anos_con_prestamo_A
	#derrames[i]*income_x_intercepts[i] for i in range(len(income_x_intercepts))]
	for i in range(len(derrames_totales)):
		message = (f'{labels[i]} tapar: {derrames[i]:.2f}/mes ({derrames_totales[i]:.0f}/tot)')
		#st.write(message)

	#st.write('─────────────────────')
	for i in range(len(opors)):
		#st.write(f'Costo de oportunidad: {opors[i]:.0f}')	
		pass
	#for i in range(len(costos_de_oportunidad)):
	#	st.write(f'Costo de oportunidad: {costos_de_oportunidad[i]:.0f}')	
	#st.write(f'Costo de oportunidad A1: {costo_oportunidad_A1:.0f}')


	#print(derrames_totales)
	# dinero pagado por nosotros

	# pulldown menu: A_pes, ...

	# Graph debt remaining
