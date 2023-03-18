# DNA Capital
# Uventet innsikt.
# Optimalt roi-tidspunkt — projeksjon + justert/funnet underveis. 
# Måte å tenke på. 
# Feltarbeid, undersøkelser, verktøy/analyse. 
# Add: • Curve for cumulate equity. • Compare with alt. opor. 
# Si la peor situación es rentable, ... 
# Risk slope
# Risk factor (point on curve)
# 	data from real estate sales? MC
# compliance probability (investing the same amount in a non-house-buying situation)
# interés variable
# mc sim on: interest, ingresos, occupancy, stock market
# problem with (i+1) vs (i+i) in some series
# Chain SimA -> SimB
# Curva de aplanamiento
# Remember that intercept 0 is the first month, intercept 30 is month 31, etc.
# Saldar+RecCap
# For display, remember to shift everything one step, so that month 0 becomes 1.
import streamlit as st; st.set_page_config(layout="wide")
import streamlit as st; #st.set_page_config(layout="wide")
import numpy as np 
import pandas as pd; #import get_data
import userinput as userinput
import process as proc
import get as get
import display as disp
from CONSTANTS import *
import gentools
from gentools import list_to_1d_matrix as to_1d
from gentools import list_to_1d_flat as to_1d_flat
import time; start_time = time.time()

disp.streamlit_hide(st.markdown)
ALERT = None
NN = {}; _ = [NN.update({ n: n }) for n in range (N)]

SuperLeft, SuperRight = st.columns([1, 1])
Columns, colGraph = userinput.get_columns(N)
Columns2, colGraph2 = userinput.get_columns(N)
sidebar = st.sidebar
st.write("  \n  "); st.write("  \n  "); st.write("  \n  "); 
data_tables = st.expander("Data tables", expanded=True)

file = "save.csv"
inputs = get.read_data(file)[0:N]

colnames = [inputs[i]['name'] for i in range(N)]
inversiones = [inputs[i]['inversion'] for i in range(N)]
tasas = [inputs[i]['tasa'] for i in range(N)]
retrasos = [inputs[i]['retraso'] for i in range(N)]
valorizaciones = [inputs[i]['valorizacion'] for i in range(N)]
max_desembolsos = [inputs[i]['max_desembolso_mensual'] for i in NN] # max_desembolso_mensual = [inputs[n]['max_desembolso_mensual'] for n in range(N)]

for i in range(N):
	with Columns2[i]:
		key = "Sim" + str(i)
		#invisibles = [inputs[n][hide_graph] for n in NN]
		invisible = st.checkbox("Deact.", value=inputs[i]['hide_graph'], key=f'hide_graph_{key}', on_change=None, disabled=False)
		if invisible == True:
			hide_graph = False
		elif invisible == False:
			hide_graph = True
		inputs[i].update({ 'hide_graph': hide_graph })

with Columns2[0]:
	#chain = st.checkbox("Cadena!", value=False, key='cadena', on_change=None, disabled=False)
	chain = False
	pass
with Columns2[1]:
	meses_boton = False
	#meses_boton = st.checkbox("Long view", value=False, key='longview', on_change=None, disabled=False)
	meses_display = 120 if meses_boton == False else MAX_MESES_INV
	pass

for i in range(N):
	with Columns[i]:
		key = "Sim" + str(i)
		inputs[i].update({ 'name': st.text_input('', value=inputs[i]['name'], key=key, disabled=inputs[i]['hide_graph'])})
		inputs[i].update(userinput.input_cols(inputs[i], chain))

with sidebar:
	st.subheader("SM Invs.  \n  ")
	expanders = [st.expander(label=f"{inputs[i]['name']}") for i in range(N)]
	for i in range(N):
		with expanders[i]:
			inputs[i].update(userinput.input_sidebar(inputs[i]))

	#_ = [inputs[n].update({ 'max_desembolso_mensual': st.number_input('Max desembolso mensual', 0, 30, max_desembolsos[n], key=f'maxdes_{inputs[n]["name"]}')}) for n in NN]
	#st.write("BBB ", [inputs[n]['max_desembolso_mensual'] for n in NN])
	expander_desembolsos = st.expander(label="Max desembolso mensual", expanded=False)
	with expander_desembolsos:
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
	#meses_display = MAX_MESES_INV

	meses_boton = st.checkbox(f"Visualización: {MAX_MESES_INV} meses", value=False, key='longview', on_change=None, disabled=False)
	meses_display = SHORT_VIEW if meses_boton == False else MAX_MESES_INV

	# Curva de aplanamiento colombiana
	curva_si_o_no = st.checkbox("Curva de aplanamiento", value=False, key='curva', on_change=None, disabled=False)
	curva_min_disabled = False if curva_si_o_no == True else True
	curva_min = 1 + (st.slider("Valor bajo", -5, 5, 1, 1, key='curva_min', disabled=curva_min_disabled) / 100)
	curva_min = np.e**(np.log(curva_min)/12)

	#st.markdown("""---""")
	expanderOpor = st.expander(label="Inversión alternativa", expanded=False)
	with expanderOpor:
		opor_cap_ini = st.number_input('Inversión alt. cap.', 0.0, 1000.0, key='opor_cap_ini')
		#opor_cap_ini = st.number_input('Inversión alt.', 0, 100, key='opor_cap_ini')
		opor_saving = st.number_input('Inversión alt.', 0, 100, key='opor_saving')
		r_mes_opor = st.slider("Crec. opor. alt.", -5, 10, 0, 1)
		r_mes_opor = 1 + (r_mes_opor/100)
		r_mes_opor = np.e**(np.log(r_mes_opor)/12)
		opor_until = st.slider("Opor.: Meses", 0, 240, 120, 1)


# Repeatedly used input variables
retrasos    = [inputs[n]['retraso'] for n in NN]
ingresos    = [inputs[n]['ingreso_pesimista'] for n in NN]
desembolsos = [inputs[n]['max_desembolso_mensual'] for n in NN] if chain==False else [processed[n]['max_desembolso_mensual'] for n in NN]
inversiones = [inputs[n]['inversion'] for n in NN]
rates       = [inputs[n]['r_mes'] for n in NN]
tasas       = [inputs[n]['tasa'] for n in NN]
caps        = [inputs[n]['cap_ini'] for n in NN]
shifts      = [inputs[n]['shift'] for n in NN]
names       = [inputs[n]['name'] for n in NN]
hides       = [inputs[n]['hide_graph'] for n in NN]

Principals            = [inversiones[n]-caps[n]-retrasos[n]*desembolsos[n] for n in NN] # proc.get_and_filter_principals(inversiones, caps, retrasos, desembolsos, ingresos)
amorts_totals_sharp   = []
amorts_totals_closest = [] # Closest above.
amorts_repay_times    = [] # Closest above.
amorts                = desembolsos
initial_diffs         = [inversiones[n]-caps[n] for n in NN]
interests             = []

zeros = [0 for _ in NN]

# Ranger-matrices for internal calculations; not graphing. 
#curva_si_o_no = False
(curva_steps, curva_next) = proc.curva_steps(N, curva_min, rates, MAX_LENGTH, periods=CURVA_TIME) if curva_si_o_no==True else (None, None)

# Four matrices to rule them all.            				# These must be shifted around within functions according to 'retrasos' and 'shifts,' but should be kept "raw" in this main script. 
# Matrices for internal calcs; not graphs.
# 1 - 3:
growth_matrix     = proc.growth_matrix(inversiones, rates, MAX_LENGTH, curva_flag=curva_si_o_no, curvas=curva_steps, curva_next=curva_next)
income_matrix     = proc.income_matrix(ingresos, MAX_LENGTH, retrasos=retrasos)
income_raw        = proc.income_matrix(ingresos, MAX_LENGTH)
#st.write("income_matrix", income_matrix[0:1])
desembolso_matrix = proc.income_matrix(desembolsos, MAX_LENGTH)
#st.write("income_matrix", income_matrix[0:1])
#equity = growth_matrix - to_1d(inversiones) + to_1d(caps) + desembolso_matrix ~~~ g+i-d
#cutoff, y = proc.matrices_intersect_greater_equal(reference=growth_matrix, test=equity)
#cutoff = [cutoff[n]+1 for n in range(len(cutoff))]
#equity = proc.cut_after_x_pos(equity, positions=cutoff)
#st.write("equity", equity)

loan_ranger           = proc.loan_ranger(Principals, tasas, length=MAX_LENGTH) 
out_of_bounds, list_  = proc.out_of_bounds_check(reference=income_matrix[0:N,-1]+desembolso_matrix[0:N,-1], test=loan_ranger[0:N,-1])
#loan_ranger           = proc.shift_sequences(loan_ranger, retrasos)
#st.write("loan", loan_ranger[0:1])
#st.write("in", income_matrix+desembolso_matrix)[0:1]; exit()
xs_repay, ys_repay    = proc.matrices_intersect_greater_equal(reference=loan_ranger, test=income_matrix+desembolso_matrix)
if out_of_bounds == True:
	for i in range(len(list_)):
		if list_[i] == True:
			xs_repay[i] = MAX_LENGTH-1 - retrasos[i] # This is arbitrary. The poition should be out of bounds, because the debt will never be repayed.
			ys_repay[i] = loan_ranger[i][-1]
	ALERT = None if out_of_bounds == False else f"⚠️ The projection is out of bounds. ⚠️  \nIn {MAX_LENGTH} months you will not repay the debt. Try increasing income or spending or try reducing interest rate."
	with colGraph:
		st.write(ALERT); st.write("See here: ", xs_repay[0], ys_repay[0])
	exit()
if out_of_bounds == 999:
	MAX_LENGTH = MAX_EXTREME
	(curva_steps, curva_next) = proc.curva_steps(N, curva_min, rates, MAX_LENGTH, periods=CURVA_TIME) if curva_si_o_no==True else (None, None)
	growth_matrix     = proc.growth_matrix(inversiones, rates, MAX_LENGTH, curva_flag=curva_si_o_no, curvas=curva_steps, curva_next=curva_next)
	income_matrix     = proc.income_matrix(ingresos, MAX_LENGTH, retrasos=retrasos)
	desembolso_matrix = proc.income_matrix(desembolsos, MAX_LENGTH)
	loan_ranger           = proc.loan_ranger(Principals, tasas, length=MAX_LENGTH) 
	out_of_bounds, list_  = proc.out_of_bounds_check(reference=income_matrix[0:N,-1]+desembolso_matrix[0:N,-1], test=loan_ranger[0:N,-1])
	ALERT = None if out_of_bounds == False else f"⚠️ The projection is out of bounds. ⚠️  \nIn {MAX_LENGTH} months you will not repay the debt. Try increasing income or spending or try reducing interest rate."

# Calc quickest payback time of debt and what the debt will be for that speed.

amorts_totals_sharp   = [loan_ranger[n][xs_repay[n]] for n in NN] # PROBLEM IF OUT OF BOUNDS
amorts_totals_closest = [ys_repay[n] for n in NN]
amorts_repay_times    = [xs_repay[n]+1 for n in NN]
interests             = [amorts_totals_sharp[_]-Principals[_] for _ in NN]
#del Principals
#del loan_ranger


#st.write("DES", DESembolso_matrix_[0:1])
# 4: Debt
pre_debt  = to_1d(initial_diffs) - desembolso_matrix
pre_debtc = pre_debt.copy()
pre_debt  = proc.set_value_after_pos(pre_debt, retrasos, value=0) # Not necessary, because of cut_tail? Also, 'retrasos' can be too few steps: last steps may be below 0.
#st.write("pre_debt", pre_debt[0:1])
bank_debt = to_1d(amorts_totals_sharp) - (income_raw + desembolso_matrix) # desembolso shifted? ***********************************
bank_debts= bank_debt[:,0]
bank_debt = proc.shift_sequences(bank_debt, retrasos, fill_values=0) # bank_debt.copy() ?
#st.write("bank_debt", bank_debt[0:1])
debt      = pre_debt + bank_debt
debt      = proc.cut_tail(debt, limits=[STRANGE_NEAR_ZERO for _ in NN], fill=0) # 0 is better, but gives strange values in debt (then later) patri matrices.
#print(debt[0:1,0:36]); exit()
#del pre_debt, bank_debt

# stack or mask 2 np matrices

#debt_matrix = proc.prepend(debt_matrix, [Principals[n] for _ in NN], until_positions=amorts_repay_times)
#debt_matrix[0][:amorts_repay_times[0]] = 0
#st.write("debt_MX", debt_matrix)
#print("debtmx: ", debt_matrix[0][:30]); exit()

# pass function?

#st.write("income_matrix", income_matrix.copy(), amorts_repay_times, fill_values=0)

# Matrices for graphing.                                income_raw ??
patrimonio_bruto = growth_matrix + proc.shift_sequences(income_matrix.copy(), amorts_repay_times, fill_values=0) # Fix so that it includes income_matrix after bank repay. 
#st.write("Repaytimes", amorts_repay_times)

#print("-+", -bank_debt[0:1,0:36] + patrimonio_bruto[0:1,0:36])
patrimonio_neto  = patrimonio_bruto - debt #growth_matrix + income_matrix - debt #- diff_matrix

#patrimonio_neto  = proc.set_None_after_x_pos(patrimonio_bruto.copy() - debt_matrix.copy(), positions=amorts_repay_times).copy()
patri_matrix = np.zeros([N*2+1, MAX_LENGTH])
patri_matrix[0:N] = patrimonio_bruto
patri_matrix[N:N*2] = patrimonio_neto # equity

patri_matrix[N*2:N*2+N_extra] = None # Alternative equity.


#st.write("patri neto", patrimonio_neto[0:1])


xs_repay = proc.to_dict(xs_repay)


####################################################################################################
#
# Use loan_cost_ranger for roi matrix! The sooner you sell, the less you lose.
#
#loan_matrix = proc.shift_sequences(loan_ranger.copy(), shifts=[inputs[n]['shift'] for n in NN])

#loan_matrix = proc.shift_sequences(loan_matrix, shifts=[inputs[n]['retraso'] for n in NN], fill_values=None)
#l_matrix = proc.repeat_last_after(loan_matrix, targets=repay_amounts)
#roi_matrix = proc.roi_matrix(inputs, mmatrix[0:N,:], temp_matrix, l_matrix)


# ROI in % 
# ROI crude
spent = (to_1d(desembolsos)*np.arange(1, MAX_LENGTH+1)) + to_1d(caps)
spent = gentools.filter_matrix_greater_than(spent, limits_list=inversiones, in_place_list=inversiones)

costmx = proc.shift_sequences(loan_ranger[::].copy(), retrasos, fill_values=0) - to_1d(Principals)
costmx[costmx < 0] = 0

#spending_times = [amorts_repay_times[_] + retrasos[_] for _ in NN]
#spent = proc.set_value_after_pos(spent, spending_times, value=200) # Use last value on repeat instead.
roi_matrix  = patrimonio_bruto[::] / (-pre_debtc + spent[::] + costmx) # Implement loan_cost_ranger

numerator   = patrimonio_bruto[::] - pre_debtc[::] - costmx[::]
denominator = spent[::]

roi_matrix  = numerator / denominator

#y_r = [2 for n in range(roi_matrix.shape[0])]
#x_r = {}

# ROI - comisiones
com = comision_constructora + comision_inmob
#roi_com_matrix = proc.roi_matrix_minus_commission(com, inputs, growth_matrix, patri_matrix[0:N,:], temp_matrix, l_matrix)
roi_com_matrix = proc.roi_minus_commission(com, growth_matrix[0:N], patrimonio_neto, spent)

#roi_com_matrix[roi_com_matrix < 0] = STRANGE_NEAR_ZERO # Is this a good idea?

# Makes little sense: Best time is month 1, but it also creates very little money in absolute terms. Plus, it takes time to find properties.
rmx              = roi_matrix.copy()
rmx[rmx < 1]     = 1
roi_earliest     = proc.optimal_max_x(rmx, start_point=1) 
roi_max          = proc.y_values_for_x(roi_matrix, roi_earliest)
rmx              = roi_com_matrix.copy()
rmx[rmx < 1]     = 1
roi_earliest_com = proc.optimal_max_x(rmx, start_point=1)
roi_max_com      = proc.y_values_for_x(roi_com_matrix, roi_earliest_com)
x_r 			 = gentools.to_dict(roi_earliest, value_type=float)
y_r 			 = roi_max
gain_roiearl_com = [patrimonio_bruto[n][pos]-inversiones[n]-com*patrimonio_bruto[n][pos] for n, pos in zip(range(N), roi_earliest_com)] # Then: gain_roiearl_com = [gain_roiearl_com[n][0] for n in NN] # Flatten
gains_at_entrega = [patrimonio_bruto[n][r]-inversiones[n]-com*patrimonio_bruto[n][r] for n, r in zip(range(N), retrasos)]
#st.write("x_r", x_r, "y_r", y_r, "hides", hides)


#cash_until_repayment_dates = [caps[n] + amorts_totals_sharp]

#ys_repay = proc.to_dict(ys_repay)
#y_s = [amorts_totals_closest[_] for _ in NN]

# Saldar con bbanco
#smatrix, x_s = proc.smatrix(income_matrix[::] + D_matrix[::], ys_repay)
#smatrix, x_s = proc.smatrix(bank_debt, y_s)
smatrix = 0 - bank_debt
smatrix[smatrix > 0] = 1
indices = smatrix.argmax(axis=1)
indices = [int(i) for i in indices]
indices = [indices[n]+shifts[n] for n in NN]
x_s = gentools.to_dict(indices)
smatrix[smatrix == 1] = None # Shaves off after.
smatrix[smatrix == 0] = None # Shaves off before.

#smatrix[smatrix < to_1d(initial_diffs)] = 999

y_s = amorts_totals_closest
#y_s = proc.to_dict(amorts_totals_closest)
#y_s = [processed[n]['loan_repay_amount'] for n in NN]

# Restaurar cap_ini para nueva compra
#duo_matrix, x_duo, y_duo = proc.duo_matrix(spent, limits=caps, end_values=amorts_totals_closest, maxlength=MAX_LENGTH)
#duo_matrix, x_duo, y_duo = proc.duo_matrix(inputs, N, NN, income_matrix, D_matrix, y_s, MAX_LENGTH)
duo_matrix = np.zeros([income_matrix.shape[0]*2, income_matrix.shape[1]])
duo_matrix[0:N] = income_matrix + desembolso_matrix - to_1d(y_s)
duo_matrix[N:N*2] = duo_matrix[0:N]
#duo_matrix[0:N] = duo_matrix[0:N] - to_1d(y_s)
duo_matrix[0:N] = proc.set_value_after_pos(duo_matrix[0:N], positions=amorts_repay_times, value=None)
duo_matrix[N:N*2] = proc.set_value_before_pos(duo_matrix[N:N*2], positions=amorts_repay_times, value=None)
caps2 = caps.copy()
caps2.extend(caps)
for n in range(N, N*2):
	m = duo_matrix[n:n+1].view()
	m[m > caps2[n]] = None

#duo_matrix[N:N*2] = proc.set_value_before_pos(duo_matrix[N:N*2], positions=amorts_repay_times, value=None)

x_duo = [0 for _ in NN] + [amorts_repay_times[n] + retrasos[n] for n in NN]
#st.write("x_duo", x_duo);
x_duo = proc.to_dict(x_duo) #; x_duo = [None for _ in NN]
y_duo = proc.to_dict(caps2)

#cutoffs, y = proc.matrices_intersect_greater_equal(reference=duo_matrix[N:N*2], test=np.zeros([1, duo_matrix.shape[1]]))
#x_duo = [cutoffs[n]+1 for n in range(len(cutoffs))]
xx, yy = proc.x_intercepts_for_y(duo_matrix[N:N*2], targets=caps) # Not implemented yet
#print(xx); 
x_duo = xx#[xx[n]+1 for n in range(len(xx))]

x_duo = proc.to_dict([amorts_repay_times[n] + int(caps[n]/(desembolsos[n]+ingresos[n])) for n in NN])




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
	x_s, y = proc.x_intercepts_for_y(smatrix, targets=zeros)
	smatrix = proc.cut_after(smatrix, targets=zeros)

	#mmatrix = proc.shift_sequences(mmatrix, shifts=cadena)
	#x_m = proc.x_intercepts_for_y(mmatrix[0:N], targets=y_m)




# Net gains
pmatrix, x_p, y_p = proc.pmatrix2(growth_matrix, income_matrix, spent, to_1d(inversiones), to_1d(amorts_totals_closest), amorts_repay_times, retrasos, shifts, out_of_bounds_value=OUT_OF_BOUNDS_VALUE)
#pmatrix = growth_matrix - to_1d(inversiones) + income_matrix - to_1d(amorts_totals_closest)
#y_p = [spent[n][amorts_repay_times[n]+retrasos[n]] for n in NN] # Todo el dinero gastado
#x_p, y = proc.x_intercepts_for_y(pmatrix, targets=y_p, out_of_bounds_value=-99)
#x_p = proc.to_dict([int(x_p[n])+shifts[n] for n in NN])

# Oportunidad alternativa (bolsa)
if r_mes_opor != 1:
	#opor_matrix = proc.opor_sequence(inputs, r_mes_opor, MAX_LENGTH, untils=x_duo) # opor_sequence2 involves eternal saving; oporsequence until recovered cap_ini
	#opor_matrix = proc.opor_sequence2(inputs, r_mes_opor, MAX_LENGTH, until=opor_until) # opor_sequence2 involves eternal saving; oporsequence until recovered cap_ini
	opor_matrix = proc.opor_sequence3(opor_cap_ini, opor_saving, r_mes_opor, MAX_LENGTH, until=opor_until) # opor_sequence2 involves eternal saving; oporsequence until recovered cap_ini
	opor_matrix = proc.shift_sequences(opor_matrix, shifts=[inputs[n]['shift'] for n in NN])
	h = patri_matrix.shape[0]
	patri_matrix[h-1] = opor_matrix[0]
	#mmatrix[5] = opor_matrix[1]
	#mmatrix[6] = opor_matrix[2]
	#mmatrix[7] = opor_matrix[3]
	patri_matrix[N*2:N*2+N_extra] = opor_matrix[0]



####################################################################################################
#																								   #
# Visuals																						   #
#                                                                                                  #
####################################################################################################

# Crop matrix
patri_matrix = patri_matrix[:,:meses_display+1]
double_shifts = shifts.extend(shifts)
patri_matrix[0:N] = proc.shift_sequences(patri_matrix[0:N], shifts=shifts)
patri_matrix[N:N*2] = proc.shift_sequences(patri_matrix[N:N*2], shifts=shifts)
roi_matrix = roi_matrix[0:N,0:int(.5*(meses_display+1))]
roi_com_matrix = roi_com_matrix[0:N,0:int(.5*(meses_display+1))]
pmatrix = pmatrix[0:N,0:max(x_p.values())+6]
pmatrix = proc.shift_sequences(pmatrix, shifts=shifts)
smatrix = smatrix[0:N,0:proc.len_longest_graph(smatrix)+11]
smatrix = proc.shift_sequences(smatrix, shifts=shifts)
duo_matrix = duo_matrix[0:N*2,0:proc.len_longest_graph(duo_matrix)+11]

# Graphs
with data_tables:
	#eln_matrix = np.e**(np.log(roi_matrix)/np.arange(1,MAX_LENGTH+1))
	#disp.show_tables(hides, { 'costmx': costmx, 'loan_ranger': loan_ranger, 'pre_debt': pre_debt, 'bank_debt': bank_debt, 'debt': debt, 'spent': spent, 'income_matrix': income_matrix, 'patrimonio_bruto': patrimonio_bruto, 'neto': patrimonio_neto, 'roi': roi_matrix, 'roi_com_matrix': roi_com_matrix, 'pmatrix': pmatrix})
	pass

with st.sidebar:
	if st.button("Save"):
		if chain == True:
			st.warning('Unchain first!', icon="⚠️")
		else:
			data_to_file = pd.DataFrame(inputs)
			data_to_file.to_csv("save.csv", index=True)

#with SuperLeft:
with colGraph:
	if ALERT != None:
		st.write(ALERT)

	tab1, roimatrx, roi2matrix, Banco, BancoCap, PagaSolo, datasheet = st.tabs(['Patrimonio', '|  ROI',  '|  ROI-Com', '|  Banco', '|  Banco+Cap','|  Ganancia neta','|  DataSheet']) #PagaSolo
	with tab1:
		xs_inv, ys_inv = proc.find_x_intercepts_for_y(patri_matrix[0:N], targets=interests)

		#names = []
		#_ = [names.append(inputs[n]['name']) for n in NN]
		_ = [names.append(inputs[n]['name'] + str(' - neto')) for n in NN]
		#names = [inputs[n]['name'] for n in NN]
		names.append("Opor")
		#names.append("Opor2")
		#names.append("Opor3")
		#names.append("Opor4")
		hides = [inputs[n]['hide_graph'] for n in NN]
		hides.extend(hides)
		hides.append(False) # For opoe sequence.
		#hides.append(False)

		x_m = proc.to_dict([meses_display for _ in range(N*2)])
		y_m = proc.to_dict([int(patri_matrix[_][-1]) for _ in range(N*2)])
		# Or:
		#y_m = (patri_matrix[0:N*2, 0:meses_display])[:,-1]
		#y_m = proc.to_dict(y_m, int_=True)
		#x_m = proc.x_intercepts_for_y(patri_matrix[0:N], targets=y_m)
		
		disp.plot2(patri_matrix, x_m, y_m, N+2, names=names, hides=hides, show_labels=True, labels=y_m, message="Patrimonio neto en color claro.")

	with roimatrx:
		message = "Retorno al vender en mes X."
		disp.plot2(roi_matrix, x_r, y_r, N, names=names, hides=hides, show_labels=True, labels=x_r, message="roi")
		st.write(message)

	with roi2matrix:
		x = proc.to_dict(roi_earliest_com)
		y = proc.to_dict(roi_max_com)
		disp.plot2(roi_com_matrix, x, y, N, names=names, hides=hides, show_labels=True, labels=x, message="roi")
		st.write("Retorno óptimo ajustado por comisiones.")

	with Banco:
		disp.plot(inputs, smatrix, x_s, zeros, labels=x_s)
		st.write("Tiempo requerido para saldar cuentas con el banco.")

	with BancoCap:
		#disp.plot2(duo_matrix, x_duo, y_duo, N=4, names=names, hides=hides, show_labels=True, labels=y_duo, message="Bcap plot2")
		disp.plot_duo(inputs, duo_matrix, x_duo, y_duo, labels=x_duo)
		st.write("Recuperar capital inicial, incl. desembolsos: desde inicio (desde hoy).")

	with PagaSolo:
		disp.plot(inputs, pmatrix, x_p, y_p, labels=x_p)
		st.write("Recuperación total [valorización+ingresos-todos los dineros gastados].")

	with datasheet:
		datasheet = proc.datasheet(colnames, inversiones, roi_matrix, roi_max, roi_earliest, roi_earliest_com, roi_max_com, gain_roiearl_com, gains_at_entrega, Principals, bank_debts, caps, interests, desembolsos, retrasos, amorts_repay_times, amorts_totals_closest) # Dict of dicts. 
		df = pd.DataFrame.from_dict(datasheet)
		st.write(df)

end_time = time.time(); print("Time to calculate: ", end_time-start_time); st.write("Time to calculate: ", end_time-start_time)