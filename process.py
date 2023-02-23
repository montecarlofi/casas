import math
import numpy as np
import streamlit as st
import pandas as pd

MONEY_MULTIPLIER = 1000

def get_columns(N):
	#Columns = []
	#a, a0, b, b0, c, c0, d = st.columns([2, 1, 2, 1, 2, 1, 2])
	#Columns.append(a); Columns.append(b); Columns.append(c); Columns.append(d); 
	#colGraph = 0
	#return Columns, colGraph
	# Works:
	colGraph, cola00, cola, cola0, colb, colb0, colc, colc0, cold = st.columns([9, 1, 2, 1, 2, 1, 2, 1, 2])
	Columns = []
	Columns.append(cola)
	Columns.append(colb)
	Columns.append(colc)
	Columns.append(cold)
	return Columns, colGraph
	# Works:
	colsizes = []
	_ = [(colsizes.append(2), colsizes.append(1)) for x in range(N)]
	colsizes.append(np.array(colsizes).sum())
	cs = st.columns(colsizes)
	Columns = []
	for i in range(0, len(colsizes)-2, 2):
		Columns.append(cs[i])
	colGraph = cs[len(colsizes)-1]
	return Columns, colGraph

def input_sidebar(inn):
	#inn = { 'name': name }
	name = inn['name']

	disabled = True if inn['hide_graph'] == True else False

	inversion = st.number_input('Monto', value=inn['inversion'], key=f'inversion_{name}', disabled=disabled)
	cap_propio = st.number_input('Cap propio', value=inn['cap_ini'], key=f'cap_propio_{name}', disabled=disabled)
	if cap_propio > inversion:
		cap_propio = inversion
	tasa = st.number_input('Tasa bancaria', value=inn['tasa']*100, key=f'tasa_{name}', disabled=disabled) / 100
	porcion = cap_propio/inversion

	inn.update({'inversion': inversion, 
		'tasa': tasa, 
		'cap_ini': cap_propio, 
		#'hide_graph': hide_graph
		#'capital': capital,
		})#'max_desembolso_mensual': max_desembolso_mensual})
		#, 'porcion': porcion, 'principal': principal, 'anos_con_prestamo': anos_con_prestamo, 'meses_con_prestamo': meses_con_prestamo, 'amort_mes': amort_mes, 'amort_total': amort_total, 'deuda_y_capital': deuda_y_capital, 'costo_prestamo': costo_prestamo, 'max_desembolso_mensual': max_desembolso_mensual})
	return inn

#def input_cols(name, ingreso=0, retraso=0, valorizacion=0, max_desembolso_mensual=0, hide_graph=False):
def input_cols(inn):
	#inn = { 'name': name }
	#st.write(name)
	disabled = True if inn['hide_graph'] == True else False

	R_high = int(inn['r_high'])
	#R_high = st.slider("High valoriz.", -5, 15, R_high, 1, key=f"high_{inn['name']}", disabled=disabled)
	valorizacion = int(inn['valorizacion'])
	valorizacion = st.slider("Valorización", -5, 15, valorizacion, 1, key=f"valorizacion_{inn['name']}", disabled=disabled)
	if valorizacion != 0:
		v = 1 + (valorizacion/100) # Buggy: 0 becomes 1 ?
		r_mes = np.e**(np.log(v)/12)
	else:
		r_mes = 1

	if R_high != 0:
		v = 1 + (R_high/100) # Buggy: 0 becomes 1 ?
		r_high = np.e**(np.log(v)/12)
	else:
		r_high = 1

	key = "ingreso_optimista_" + str(inn['name'])
	value = int(inn['ingreso_pesimista']*MONEY_MULTIPLIER)
	ingreso_pesimista = st.slider("Ing pesimista K", 0, 8000, value, 100, key=key, disabled=disabled)
	ingreso_optimista = 0#st.slider("Ingreso optimista K", 0, 5000, 2500, 100)
	ingreso_pesimista /= MONEY_MULTIPLIER
	ingreso_optimista /= MONEY_MULTIPLIER

	retraso = st.slider("Proyecto meses", 0, 36, int(inn['retraso']), 6, key=f'retraso_{inn["name"]}', disabled=disabled)
	#shift = st.slider("Mover", 0, 96, int(inn['shift']), 1, key=f'shift_{inn["name"]}', disabled=disabled)
	shift = st.number_input('Mover', 0, 240, int(inn['shift']), key=f'shift_{inn["name"]}', disabled=disabled)

	#hide_graph = st.checkbox("Hide", value=hide_graph, key=f'visible_{name}', on_change=None, disabled=False)
	#st.write("rate ", r_mes**12)
	inn.update({'name': inn['name'],
		'valorizacion': valorizacion, 
		'r_mes': r_mes, 
		#'r_high': R_high, 
		'ingreso_pesimista': ingreso_pesimista, 
		'ingreso_optimista': ingreso_optimista, 
		'hide_graph': inn['hide_graph'], 
		#"max_desembolso_mensual": max_desembolso_mensual,
		'retraso': retraso,
		'shift': shift
		})
	
	return inn


def sim0(inv, cap, costo_prestamo, ingreso, r, n, meses_con_prestamo):
	y = np.zeros([1, n])
	for i in range(0, n):
		#y[0][i] = -costo_prestamo + ingreso*i + inv*r**i #- inv#- cap
		y[0][i] = ingreso*i + inv*r**i #- inv#- cap
	return y

def sim_main(inputs, MAX_LENGTH, no_growth=False) -> 'mtrx':
	N = len(inputs)
	mx = np.zeros([N, MAX_LENGTH])

	for n in range(N):
		# arange
		r = inputs[n]['r_mes']
		inv = inputs[n]['inversion']
		ingreso = inputs[n]['ingreso_pesimista']
		retraso = inputs[n]['retraso']

		if retraso == 0:
			start = 0
		else:
			start = retraso

		#y = np.arange(0-retraso, MAX_LENGTH+retraso)
		#y[:] = ingreso*(i-start) + inv*r**(i-start)

		y = np.zeros([1, MAX_LENGTH])
		for i in range(0, start):
				y[0][i] = None
		if no_growth == True:
			#print("Here")
			for i in range(start, MAX_LENGTH):
				# (i+1-start) ? 
				y[0][i] = ingreso*(i+1-start)# + inv*r**(i+1)#-start)
		else:
			for i in range(start, MAX_LENGTH):
				# (i+1-start) ? 
				y[0][i] = ingreso*(i+1-start) + inv*r**(i+1)#-start)
		mx[n] = y
	return mx

def sim_income_only(inputs, MAX_LENGTH) -> 'mtrx':
	N = len(inputs)
	mx = np.zeros([N, MAX_LENGTH])

	for n in range(N):
		# arange
		ingreso = inputs[n]['ingreso_pesimista']
		retraso = inputs[n]['retraso']

		if retraso == 0:
			start = 0
		else:
			start = retraso

		#y = np.arange(0-retraso, MAX_LENGTH+retraso)
		#y[:] = ingreso*(i-start) + inv*r**(i-start)

		y = np.zeros([1, MAX_LENGTH])
		for i in range(0, start):
				y[0][i] = None
		for i in range(start, MAX_LENGTH):
			y[0][i] = ingreso*(i+1-start)
		mx[n] = y
	return mx

def sim_income_and_desembolso(inputs, MAX_LENGTH) -> 'mtrx':
	N = len(inputs)
	mx = np.zeros([N, MAX_LENGTH])

	for n in range(N):
		# arange
		desembolso = inputs[n]['max_desembolso_mensual']
		ingreso = inputs[n]['ingreso_pesimista']
		retraso = inputs[n]['retraso']

		if retraso == 0:
			start = 0
		else:
			start = retraso

		#y = np.arange(0-retraso, MAX_LENGTH+retraso)
		#y[:] = ingreso*(i-start) + inv*r**(i-start)

		y = np.zeros([1, MAX_LENGTH])
		for i in range(0, start):
				y[0][i] = None
		for i in range(start, MAX_LENGTH):
			y[0][i] = ingreso*(i+1-start) + desembolso*(i+1-start)
		mx[n] = y
	return mx

def income_matrix(inputs, length): # WORKS
	N = len(inputs)
	mx = np.zeros((N, length))
	
	mx[:] = np.arange(1, length+1)
	ings = np.array([inputs[n]['ingreso_pesimista'] for n in range(N)]).reshape(N, 1)

	mx[:] = mx[:] * ings

	for n in range(N):
		retraso = inputs[n]['retraso']
		if retraso > 0:
			mx[n:n+1,retraso:length] = mx[n:n+1,0:length-retraso]
			mx[n:n+1,0:retraso] = None
	return mx

def desembolso_matrix(inputs, length): # WORKS
	N = len(inputs)
	mx = np.zeros((N, length))
	
	mx[:] = np.arange(1, length+1)
	ings = np.array([inputs[n]['max_desembolso_mensual'] for n in range(N)]).reshape(N, 1)

	mx[:] = mx[:] * ings
	return mx

def growth_matrix(inputs, length): # WORKS
	N = len(inputs)
	mx = np.zeros((N, length))
	
	mx[:] = np.arange(1, length+1)
	rs = np.array([inputs[n]['r_mes'] for n in range(N)]).reshape(N, 1)
	invs = np.array([inputs[n]['inversion'] for n in range(N)]).reshape(N, 1)

	mx[:] = invs*rs**mx

	# This works, but not valid: Valorización durante planos también.
	#for n in range(N):
	#	retraso = inputs[n]['retraso']
	#	if retraso > 0:
	#		mx[n:n+1,retraso:length] = mx[n:n+1,0:length-retraso]
	#		mx[n:n+1,0:retraso] = None

	return mx

def growth_matrix_curva(inputs, length):
	N = len(inputs)
	mx = np.zeros((N, length))
	
	#mx[:] = np.arange(1, length+1)
	rs = np.array([inputs[n]['r_mes'] for n in range(N)]).reshape(N, 1)
	r_highs = np.array([inputs[n]['r_high'] for n in range(N)]).reshape(N, 1)
	invs = np.array([inputs[n]['inversion'] for n in range(N)]).reshape(N, 1)
	print("rs ", rs, " r_highs ", r_highs)
	slopes_10_years = [np.e**(np.log(rs[i]/r_highs[i])/120) for i in range(N)] # Downtrend if high is higher than r_mes.
	slopes_10_years = np.array(slopes_10_years).reshape(N, 1)
	s = slopes_10_years
	slope_matrix = np.zeros([N, 120])

	slope_matrix[:,0] = slopes_10_years[0]
	for i in range(1, 120):
		slope_matrix[:,i] = slope_matrix[:,i-1] * s[:,0] # This is the growth/decline of the rate itself.

		#slope_matrix[0:N,i*12:(i+1)*12] = slopes_10_years[i]
	print(slope_matrix)

	rates = np.ones([N, 120])
	rates[:] = rs[:]

	mx = np.zeros([N, length])
	mx[:,0] = invs[:,0]*slope_matrix[:,0]
	for i in range(1, 120):
		mx[:,i] = mx[:,i-1] * slope_matrix[:,i]
	#print("mx ", mx[:,119]); exit()
	ranger = np.arange(1, -120+length+1)
	mx[:,120:length-120] = mx[:,119] * rs[:] ** ranger[:]


	return mx

def curva_matrix(curva_de_aplanamiento, N): # Returns N x 120 matrix UNFINISHED
	curva = curva_de_aplanamiento
	mx = np.zeros([N, len(curva)])

	r_seq = np.zeros([N, length]) # Seq (1-dim matrix) or matrix?
	for i in range(len(curva_de_aplanamiento)):
		c = curva_de_aplanamiento[i]
		c = np.e**(np.log(1+c/100)/12)
		p = (i+1)*12
		r_seq[0:N,i*12:p] = c
	r_seq[0:N,p:] = rs
	mx[::] = invs*r_seq**mx

def opor_sequence(inputs, untils, r, length):
	# a * (1-r**N) / (1-r)
	N = len(inputs)
	mx = np.zeros((N, length))
	if r == 1:
		mx[::] = None
		return mx[::]

	ds = np.array([inputs[n]['max_desembolso_mensual'] for n in range(N)]).reshape(N, 1)
	#rs = np.array([inputs[n]['r_mes'] for n in range(N)]).reshape(N, 1)
	rs = np.array([r for _ in range(N)]).reshape(N, 1)
	invs = np.array([inputs[n]['cap_ini'] for n in range(N)]).reshape(N, 1)
	#us = np.array([u for u in range(N)]).reshape(N, 1)


	ds = [inputs[n]['max_desembolso_mensual'] for n in range(N)]
	#rs = np.array([inputs[n]['r_mes'] for n in range(N)]).reshape(N, 1)
	rs = [r for _ in range(N)]
	invs = [inputs[n]['cap_ini'] for n in range(N)]
	
	mx[:] = np.arange(1, length+1)
	ranger = np.arange(1, length+1)

	for i in range(N):
		point = untils[i]#; st.write("Point ", point)
		m = mx[i:i+1,0:point]
		mx[i:i+1,0:point] = invs[i]*rs[i]**m + ds[i]*(1-rs[i]**m)/(1-rs[i])
		#mx[i:i+1,0:point] = invs*rs[i]**mx[i:i+1,0:point] + ds[i]*(1-rs[i]**mx[i:i+1,0:point])/(1-rs[i])
		capital = mx[i:i+1,point:point+1][0]
		capital = mx[i][point-1]
		#st.write("Capital ", capital)
		mx[i:i+1,point:length] = capital*rs[i]**ranger[0:length-point]
	return mx
	#mx[:] = invs*rs**mx + ds*(1-rs**mx)/(1-rs)
	#return mx

def opor_sequence2(inputs, untils, r, length): # Returns according to each desembolso, but─NOTE─does not shift by starting points.
	# a * (1-r**N) / (1-r)
	N = len(inputs)
	mx = np.zeros((N, length))
	if r == 1:
		mx[::] = None
		return mx[::]

	ds = np.array([inputs[n]['max_desembolso_mensual'] for n in range(N)]).reshape(N, 1)
	#rs = np.array([inputs[n]['r_mes'] for n in range(N)]).reshape(N, 1)
	rs = np.array([r for _ in range(N)]).reshape(N, 1)
	invs = np.array([inputs[n]['cap_ini'] for n in range(N)]).reshape(N, 1)
	#us = np.array([u for u in range(N)]).reshape(N, 1)
	
	mx[:] = np.arange(1, length+1)
	mx[:] = invs*rs**mx + ds*(1-rs**mx)/(1-rs)
	return mx

def loan_matrix(inputs, length):
	N = len(inputs)
	mx = np.zeros((N, length))

	mx[:] = np.arange(1, length+1)

	for n in range(N):
		i = inputs[n]['tasa']/12 
		P = inputs[n]['inversion'] - (inputs[n]['cap_ini'] + inputs[n]['max_desembolso_mensual']*inputs[n]['retraso'])
		#retraso=inputs[n]['retraso']
		retraso = 0
		#mx[n][:] = ((P*i*(1+i)**mx[n][:]) / ((1+i)**mx[n][:] - 1)) # Each step is the monthly amort if that step represented the months.
		mx[n][:] = ((P*i*(1+i)**(mx[n][:]-retraso)) / ((1+i)**(mx[n][:]-retraso) - 1)) * (mx[n][:]-retraso)
		#mx[n][:] = mx[n][:] - 10

	return mx

def sim_debt(inputs, xs, ys, length):
	N = len(inputs)
	mx = np.zeros((N, length))
	amorts_tot = ys
	amorts = [ys[n]/xs[n] for n in range(N)]
	#print("Amorts ", amorts)
	#print("tot ", amorts_tot)

	#mx[:] = np.arange(length, 0, -1)

	for n in range(N):
		retraso = inputs[n]['retraso']
		if retraso == 0:
			start = 0
		else:
			start = retraso

		for i in range(0, start):
			mx[n][i] = None#amorts_tot[n] - i*amorts[i]

		for i in range(start, xs[n]+retraso+1):
			mx[n][i] = amorts_tot[n] - (i-retraso)*amorts[n]

		mx[n][i:] = None

	for n in range(N):
		#retraso=inputs[n]['retraso']
		retraso = 0
		#mx[n][:] = (mx[n][:]-retraso) * (amorts_tot[n]/xs[n])
		# Shift left
	return mx

def calc_min_repay_times(inputs, loan_matrix) -> list:
	xs = {}
	ys = {}
	N = len(inputs)

	for n in range(N):
		payment = inputs[n]['max_desembolso_mensual'] + inputs[n]['ingreso_pesimista']
		for i in range(0, loan_matrix.shape[1]):
			if payment * (i+1) >= loan_matrix[n][i]:
				xs[n] = i+1
				ys[n] = loan_matrix[n][i]
				break
	return xs, ys

def shift_sequences(mtrx, shifts):
	ver = mtrx.shape[0]
	hor = mtrx.shape[1]

	#if np.array(shifts).sum() == 0: return mtrx
	flag = "donothing"
	for n in range(ver):
		if shifts[n] != 0:
			flag = "continue"
	if flag == "donothing":
		return mtrx

	new = np.zeros([ver, hor])
	for v in range(ver):
		n_steps = shifts[v]
		if n_steps > 0:
			new[v:v+1, 0:n_steps] = None
			new[v:v+1, n_steps:] = mtrx[v:v+1, 0:hor-n_steps]
		elif n_steps < 0:
			n_steps = int(np.sqrt(n_steps**2)) # Make positive.
			new[v:v+1, 0:hor-n_steps] = mtrx[v:v+1, n_steps:hor]
			new[v:v+1, hor-n_steps:hor] = None
		else:
			new[v] = mtrx[v]

	return new

def shift_right(mx, list_): # .......?............... No work. elegantly.
	N = mx.shape[0]
	w = mx.shape[1]
	shifts = np.array(list_).reshape(N, 1)

	mx[::] = mx[:,0:shifts[:]] * None + mx[:,0:w-shifts[:]]
	return mx

# Don't use 480-konstant!
def shift_left(line, n_steps):
	ver = 1
	hor =480# line.shape[1]

	#line = np.array(line)

	new = np.zeros([ver, hor])
	new[0:1, 0:hor-n_steps] = line[0:1, n_steps:hor]
	new[0:1, hor-n_steps:hor] = None
	return new

def shift_down(mx, y_values): # WORKS
	y_values = np.array(y_values).reshape(mx.shape[0], 1)
	#for i in range(mx.shape[1]):
	#	mx[:,i] = mx[:,i] - y_values
	return mx[:] - y_values

def find_x_intercepts_for_y(mtrx, targets): # WORKS
	xs = {}
	ys = {}
	ver = mtrx.shape[0]
	hor = mtrx.shape[1]

	# Rounds target number downwards.
	targets = [int(targets[number]) for number in range(len(targets))]
	
	for v in range(ver):
		for h in range(hor):
			#print(f'Is {mtrx[v][h]} > {targets[v]}?')
			if mtrx[v][h] >= targets[v]:# and math.isnan(mtrx[v][h]) == True:
				xs[v] = h
				ys[v] = mtrx[v][h]
				break
	return xs, ys 

def x_intercepts_for_y(mtrx, targets): # WORKS
	xs = {}
	ver = mtrx.shape[0]
	hor = mtrx.shape[1]

	# Rounds target number downwards.
	targets = [int(targets[number]) for number in range(len(targets))]
	
	for v in range(ver):
		for h in range(hor):
			if mtrx[v][h] >= targets[v]:# and math.isnan(mtrx[v][h]) == True:
				xs[v] = h
				break
	return xs

def to_dict(list_or_array, int_=False): # works
	d = {}
	for e in range(len(list_or_array)):
		if int_==True:
			d.update({ e: int(list_or_array[e]) })
		else:
			d.update({ e: list_or_array[e] })
	return d

def add_list_to_elements_in_matrix(mtrx, addlist):
	addmtrx = np.array(addlist)
	addmtrx = addmtrx.reshape(len(addlist), 1)
	return mtrx[:] + addmtrx

def sim_abono(amort, amort_mes, meses_con_prestamo, MAX_LENGTH, retraso=0):
	#y = np.zeros([1, meses_con_prestamo+retraso])
	y = np.zeros([1, MAX_LENGTH])

	if retraso == 0:
		start = 0
	else:
		start = retraso

	y[0:1,0:start] = None
	y[0:1,start:start+meses_con_prestamo] = [amort - amort_mes*x for x in range(meses_con_prestamo)]
	y[0:1,start+meses_con_prestamo:MAX_LENGTH] = [None for x in range(MAX_LENGTH-(start+meses_con_prestamo))]

	return y

def cut_after_longest(mtrx, extra=0): # WORKS
	longest = 0
	for j in range(mtrx.shape[0]):
		l = mtrx.shape[1] - 1
		while l > 0:
			if math.isnan(mtrx[j][l]) == True:
				l -= 1
			else:
				if longest <= l:
					longest = l
				break
	return mtrx[0:mtrx.shape[0],0:longest+extra]

def len_longest_graph(mtrx): # WORKS
	longest = 0
	for j in range(mtrx.shape[0]):
		l = mtrx.shape[1] - 1
		while l > 0:
			if math.isnan(mtrx[j][l]) == True:
				l -= 1
			else:
				if longest <= l:
					longest = l
				break
	return longest

#print(sim_abono(10, 1, 4, 3))

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

def sim_paga_solo(inv, debt_and_cap, ingreso, r, n, retraso=0, desembolso=0):
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

def sim_llegar_a_0(inn, x_intercept, y_intercept, MAX_LENGTH):
	debt_and_cap = y_intercept
	#debt_and_cap = 

	ingreso = inn['ingreso_pesimista']
	cuota = inn['max_desembolso_mensual']
	inv = inn['inversion']
	r = inn['r_mes']
	retraso = inn['retraso']

	y = np.zeros([1, MAX_LENGTH])
	for i in range(retraso):
		y[0][i] = -debt_and_cap + i*cuota
	for i in range(retraso, x_intercept):
		y[0][i] = -debt_and_cap + i*cuota + i*ingreso

	#print("i ", i)
	y[0:1,i+1:] = None
	return y

def cut(mtrx, targets, retrasos):
	m = mtrx.shape[0]
	n = mtrx.shape[1]
	#print(mtrx[0:4,0:2])
	#y = np.zeros([1, n])
	#print("retrasos ", retrasos)
	#print("m ", m)
	#for j in range(m):
	#	if retrasos[j] == 0:
	#		start = 0
	#	else:
	#		start = retrasos[j]
	#		for k in range(0, start):
	#			mtrx[j][k] = None

	for j in range(m):
		if retrasos[j] == 0:
			start = 0
		else:
			start = retrasos[j]
		for i in range(start, n): # Could go out of bounds because of i+1 and i+2 ?
			if mtrx[j][i] >= targets[j]:
				mtrx[j:1,i+1:] = None
				break
				#mtrx[j:1,i+1:] = None
	return mtrx

def cut_after(mtrx, targets) -> 'mtrx': # WORKS
	ver = mtrx.shape[0]; hor = mtrx.shape[1]

	for v in range(ver):
		for h in range(hor):
			if mtrx[v][h] >= targets[v]:
				mtrx[v][h:] = None
				#mtrx[v][h] = None
				break
	return mtrx

def cut_before(mtrx, targets) -> 'mtrx':
	ver = mtrx.shape[0]; hor = mtrx.shape[1]
	for v in range(ver):	
		#for h in range(hor-1, 0-1, -1):
		for h in range(hor-1, -1, -1):
			if mtrx[v][h] < targets[v]:
				#print("less")
			#	pass
				#print(mtrx[v][h], " = ", targets[v])
				#mtrx[v][:h] = None
				mtrx[v][h] = None
				#break
	return mtrx

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

def intercepts(mtrx, targets, retrasos=0):
	xs = {}
	ys = {}
	n = mtrx.shape[1]
	m = mtrx.shape[0]

	for k in range(m):
		if retrasos[k] == 0: # Try
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
	#print(xs)
	return xs, ys 

def intercepts_llegar_a_0(inputs, MAX_LENGTH):
	N = len(inputs)
	x_intercepts = {}
	y_intercepts = {}
	amorts = [None for _ in range(N)]

	for n in range(N):
		i = inputs[n]['tasa']/12 
		P = inputs[n]['inversion']-inputs[n]['cap_ini'] #= inputs[n]['principal']#; n = meses_con_prestamo
		#mes = 0
		paid_so_far = 0
		#monthly = inputs[n]['max_desembolso_mensual'] + inputs[n]['ingreso_pesimista']

		#if inputs[n]['retraso'] > 0:
		#	paid_so_far += inputs[n]['max_desembolso_mensual']
		#else:
		#	paid_so_far += inputs[n]['max_desembolso_mensual'] + inputs[n]['ingreso_pesimista']

		subtract = inputs[n]['retraso']
		#amort_total_temp = 0

		paid_so_far = inputs[n]['retraso'] * inputs[n]['max_desembolso_mensual']

		#for k in range(1, inputs[n]['retraso']):
		#	paid_so_far += inputs[n]['max_desembolso_mensual'] #+ inputs[n]['ingreso_pesimista']
		start = 1+ inputs[n]['retraso']
		for k in range(start, MAX_LENGTH):
			P = inputs[n]['inversion']-inputs[n]['cap_ini'] - paid_so_far	

			amort_mes_temp = (P*i*(1+i)**k) / ((1+i)**k - 1)
			amort_total_temp = amort_mes_temp * k
			# total = amort_total_temp + P

			if paid_so_far >= amort_total_temp:
				#mes = k
				x_intercepts.update({ n: k + 1 }) # + 1
				y_intercepts.update({ n: amort_total_temp })
				amorts[n] = amort_mes_temp
				break

			paid_so_far += inputs[n]['max_desembolso_mensual'] + inputs[n]['ingreso_pesimista']

	#st.write(x_intercepts)
	#st.write(amorts)
		#st.write(y_intercepts)
		#if mes < MAX_LENGTH:
		#	for k in range(mes, MAX_LENGTH):

	#x_intercepts.update({0:-44})
	return amorts, x_intercepts, y_intercepts

def costos_prestamos(inputs, x_intercepts, y_intercepts): # Delme?
	st.write("here")
	N = len(inputs)
	costos_prestamos = [0 for _ in range(N)]
	for n in range(N):
		P = inputs[n]['inversion'] - inputs[n]['cap_ini'] - inputs[n]['retraso'] * inputs[n]['max_desembolso_mensual']
		i = inputs[n]['tasa'] / 12
		m = x_intercepts[n]

		amort_mes = (P*i*(1+i)**m) / ((1+i)**m - 1)
		costos_prestamos[n] = amort_mes * x_intercepts[n] - P

	return costos_prestamos
	#return [y_intercepts]

def add_dict_to_processed(processed, varname, dict_):
	for k in dict_:
		processed[k].update({
			varname: dict_[k]
			})
	return processed

def add_list_to_processed(processed, varname, list_):
	n = len(list_)
	for k in range(n):
		processed[k].update({
			varname: list_[k]
			})
	return processed	

def loans(inputs, processed):
	N = len(inputs)
	return [inputs[n]['inversion'] - inputs[n]['cap_ini'] - inputs[n]['retraso'] * inputs[n]['max_desembolso_mensual'] for n in range(N)]

def prices_of_loans(inputs, processed):
	N = len(inputs)
	cash_prior = [inputs[n]['retraso'] * inputs[n]['max_desembolso_mensual'] for n in range(N)]
	principals = [-cash_prior[n] + inputs[n]['inversion']-inputs[n]['cap_ini'] for n in range(N)] 
	print("Principals: ", principals)
	return [processed[n]['loan_repay_amount'] - principals[n] for n in range(N)]

def cash_spent_to_repay(inputs, processed):
	N = len(inputs)
	return [processed[n]['loan_repay_amount']+inputs[n]['cap_ini'] - inputs[n]['ingreso_pesimista'] * processed[n]['loan_repay_time'] ]

def cash_spent_to_recover_loan_and_capital(inputs, processed):
	return 0

def cash_spent_during_loan_repayment(inputs, processed):
	N = len(inputs)
	return [processed[n]['loan_repay_time'] * inputs[n]['max_desembolso_mensual'] for n in range(N)]

def cash_and_income_spent_during_loan_repayment(inputs, processed):
	N = len(inputs)
	return [processed[n]['loan_repay_time'] * inputs[n]['max_desembolso_mensual'] + processed[n]['loan_repay_time'] * inputs[n]['ingreso_pesimista'] for n in range(N)]

def cash_and_capital_spent(inputs, processed):
	N = len(inputs)
	return [inputs[n]['retraso'] * inputs[n]['max_desembolso_mensual'] + processed[n]['loan_repay_time'] * inputs[n]['max_desembolso_mensual'] + inputs[n]['cap_ini'] for n in range(N)]

def make_table(names, hides, mtrx, N):
	#N = mtrx.shape[0]
	data = []
	line_data = mtrx[0]
	length = line_data.shape[0]
	#print("Length ", length)
	for l in range(N):
		line_data = mtrx[l]
		for i in range(line_data.shape[0]):
			d = { 'name': names[l], 'mes': i, 'y': line_data[i], 'blob': 1 }
			if hides[l] == True:
				pass
			else:
				data.append(d)

	return pd.DataFrame(data)
