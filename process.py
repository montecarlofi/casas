import math
import numpy as np # import pandas as pd
from CONSTANTS import * 
import streamlit as st

# Curva de valorización à la colombiana.
def curva_steps(N, min_, valorizaciones, length, periods=120):
	curvas = np.ones([N, periods])
	curvas_next = np.ones([N, 1]) * min_
	for i in range(N):
		r = 1 + valorizaciones[i]/100
		r = np.e**(np.log(r)/12)
		curva_reducer = np.e**(np.log(min_/r)/periods)
		curvas[i:i+1,0:periods] = r * curva_reducer ** np.arange(1.0, periods+1)
		#curvas[i:i+1,periods:] = min_
	return curvas, curvas_next

# Accumulated income/spending.
def income_matrix(dineros, maxlength, retrasos=None):
	N = len(dineros)
	mx = np.zeros((N, maxlength))
	
	mx[:] = np.arange(1, maxlength+1)
	ings = np.array(dineros).reshape(N, 1)

	mx[:] = mx[:] * ings

	if hasattr(retrasos, "__len__") == True: # If is list
		for n in range(N):
			retraso = retrasos[n]
			if retraso > 0:
				mx[n:n+1,retraso:maxlength] = mx[n:n+1,0:maxlength-retraso]
				mx[n:n+1,0:retraso] = 0 # alt: = None
	return mx

# Exponential growth (matrix is accumulation: each step represents the cumulative value).
def growth_matrix(inversiones, rates, length, curva_flag=False, curvas=None, curva_next=0):
	N = len(inversiones)
	invs = np.array(inversiones).reshape(N, 1)
	
	mx = np.zeros((N, length))

	if curva_flag == False:
		rs = np.array(rates).reshape(N, 1)
		mx[::] = np.arange(1, length+1) # These will be the exponents.
		mx[::] = invs*rs**mx

	else: # Make step-on-step calculation for the curvas length. 
		length_ = curvas.shape[1]
		mx[:,0:1] = invs[:]*curvas[:,0:1] # **mx[:,0:1]
		for i in range(1, length_):
			mx[:,i:i+1] = mx[:,i-1:i] * curvas[:,i:i+1]
		mx[:,length_:] = np.arange(1, length-length_+1) # Exponents.
		mx[:,length_:] = mx[:,length_-1:length_] * curva_next ** mx[:,length_:]

	return mx

# Each step (x) shows what the total repayment would be if it were to be payed back in (x) number of months.
def loan_ranger(Ps, tasas, length):
	N = len(Ps)
	mx = np.zeros((N, length))

	mx[:] = np.arange(1, length+1)

	for n in range(N):
		i = tasas[n]
		P = Ps[n]
		if i == 0 or i == None or math.isnan(i) == True: # If no interest rate
			mx[n][:] = P # 0 # Not None ?
		else:
			i = tasas[n]/12 
			#mx[n][:] = (P*i*(1+i)**mx[n][:]) / ((1+i)**mx[n][:] - 1)
			mx[n][:] = ((P*i*(1+i)**(mx[n][:])) / ((1+i)**(mx[n][:]) - 1)) * (mx[n][:])
	return mx


# Returns True if at least one is out of bounds + list of trues/falses.
def out_of_bounds_check(reference, test):
	list_ = []
	all_false = [False for _ in range(reference.shape[0])]
	for i in range(reference.shape[0]): #st.write(f'Is {test[i]} >= {reference[i]}?')
		if test[i] > reference[i]:
			result = True
		else:
			result = False
		list_.append(result) #st.write("bounds check: [ref] [test] ", reference, test)
	return list_ != all_false, list_

def out_of_bounds_check_same(reference, test):
	at_least_one = False
	list_ = []
	for i in range(reference.shape[0]):
		if test[i] > reference[i]:
			at_least_one = True
			result = True
		else:
			result = False
		list_.append(result)
	return at_least_one, list_


# At what position for both and value in 
def matrices_intersect_greater_equal(reference, test) -> list: # Finds when, and for what vales, lines in one matrix intersects with lines in another. Intersection is while ma >= mb.
	ma, mb = reference, test
	xs = {} #[]
	ys = {} #[]
	N = ma.shape[0]
	H = ma.shape[1]

	for n in range(N):
		a, b = ma[n], mb[n]
		x, y = None, None # This creates void when desembolso = 0.
		#x, y = -9999, -9999 # -1, -1

		for pos in range(H):
			#(x, y) = (pos, b[pos]) if a[pos] >= b[pos] else (x, y)
			if a[pos] <= b[pos]:
				x, y = pos, b[pos]
				break
		xs[n] = x
		ys[n] = y
	#st.write("xs, ys", xs, ys)
		########st.write(a[0:40]); st.write(b[0:40])
		#print(a[0:40])
		#ab, a_index, b_index = np.intersect1d(a, b, return_indices=True)
		#xs.append(b_index)
		#print("index:", a_index)
#		index_of_first = np.argmax((a <= b) == True)
		#print("index: ", index_of_first)
		#xs.append(index_of_first)
		#ys.append(b[index_of_first])
	return xs, ys


def prepend(mtrx, values, until_positions) -> 'mtrx':
	ver = mtrx.shape[0]; hor = mtrx.shape[1]
	for v in range(ver):
		mtrx[v:v+1,0:until_positions[v]] = values[v]
	return mtrx
























#def calc_min_repay_times(loan_ranger, cash=0, chain=False) -> dict: # Returns error msg when income + max_desemb are below some limit.
def calc_min_repay_times(inputs, processed, loan_matrix, chain=False) -> dict: # Returns error msg when income + max_desemb are below some limit.
	xs = {}
	ys = {}
	N = len(inputs)

	for n in range(N):
		if chain == True:
			payment = processed[n]['max_desembolso_mensual'] + inputs[n]['ingreso_pesimista']
		else:
			payment = inputs[n]['max_desembolso_mensual'] + inputs[n]['ingreso_pesimista']
		for i in range(0, loan_matrix.shape[1]):
		#for i in range(0, MAX_LENGTH):
			#st.write(f"i: {i} ... {n} payment: ", payment * (i+1), "loan_matrix: ", loan_matrix[n][i])
			if loan_matrix[n][i] == 0: # Or 0.0 # Redundant?
				xs[n] = i
				ys[n] = loan_matrix[n][i]
				#print("Was equal"); st.write("Was equal")
				break
			elif payment * (i+1) >= loan_matrix[n][i]:
				xs[n] = i+1 # Why +1? The next value over?
				ys[n] = loan_matrix[n][i]
				break
	return xs, ys


def get_and_filter_principals(inversiones, caps, retrasos, desembolsos, ingresos):
	ps = [inversiones[n] - caps[n] - retrasos[n]*desembolsos[n] for n in range(len(inversiones))]
	return list(map(lambda x: x if x >= 0 else 0, ps))


# From max debt to 0.
def debt_mx2(totals, parts, repeats, maxlength=MAX_LENGTH):
	N = len(totals)
	mx = np.zeros([N, maxlength])

	for n in range(N):
		s = repeats[n]
		mx[n:n+1,0:s] = totals[n] - parts[n] * np.arange(1, s+1) # Same as: parts[n] * np.arange(1+s, 1, -1)
		mx[n:n+1,s-1:s] = 0 if mx[n:n+1,s-1:s] <= 0 else mx[n:n+1,s-1:s]
	return mx

def debt_mx3(totals, parts, repeats, maxlength=MAX_LENGTH):
	length = maxlength
	st.write("Totals", totals)
	N = len(totals)
	mx = np.zeros([N, length])

	for n in range(N):
		s = repeats[n]

		mx[n:n+1,s:length] = totals[n] - parts[n] * np.arange(1, length+1-s)
		#mx[n:n+1,len_-1:len_] = 0 if mx[n:n+1,len_-1:len_] <= 0 else mx[n:n+1,len_-1:len_]

		#mx[n:n+1,0:len_] = sums[n] - parts[n] * np.arange(1, len_+1)
		#mx[n:n+1,len_-1:len_] = 0 if mx[n:n+1,len_-1:len_] <= 0 else mx[n:n+1,len_-1:len_]
		#mx[n:n+1,len_:] = 0 # None
	return mx



def debt_mx(sums=None, parts=None, repetitions=None, retrasos=None, caps=None, maxlength=None):
	if sums == None:
		sums = [parts[_] * repetitions[_] for _ in range(len(parts))]
	if parts == None:
		parts = [sums[_]/repetitions[_] for _ in range(len(repetitions))]
	if repetitions == None:
		repetitions = [int(sums[_]/parts[_])+1 for _ in range(len(sums))] # +0 enough?
	#if hasattr(retrasos, "__len__") == False:
	#	retrasos = [0 for _ in range(N)]
	if caps == None:
		caps = [0 for _ in range(len(sums))]

	N = len(sums)
	mx = np.zeros([N, maxlength])

	for n in range(N):
		print(f'Reps: {repetitions[n]}, sums: {sums[n]}, parts: {parts[n]}')
		reps = repetitions[n]
		if parts[n] != 0:
			division = sums[n]/parts[n]
		else:
			division = 0
		while float(reps) > division: # Make sure to not go past 0.
			reps -= 1

		len_ = reps# repetitions[n]
		#ret = retrasos[n]
		#a = retrasos[n]
		#b = repetitions[n]
		#mx[n:n+1,0:len_] = parts[n] * np.arange(1+len_, 1, -1)
		#mx[n:n+1,0:a] = caps[n] + parts[n] * np.arange(1, a+1)
		#mx[n:n+1,a:a+b] = sums[n] - parts[n] * np.arange(1, b+1)
		#mx[n:n+1,a+b-1:a+b] = 0 if mx[n:n+1,a+b-1:a+b] <= 0 else mx[n:n+1,a+b-1:a+b]
		mx[n:n+1,0:len_] = sums[n] - parts[n] * np.arange(1, len_+1)
		mx[n:n+1,len_-1:len_] = 0 if mx[n:n+1,len_-1:len_] <= 0 else mx[n:n+1,len_-1:len_]
		mx[n:n+1,len_:] = 0 # None
	return mx




def shift_sequences(mtrx, shifts, fill_values=None): # Positive shifts right; negative, left.
	ver = mtrx.shape[0]
	hor = mtrx.shape[1]
	
	if hasattr(fill_values, "__len__") == True: # If is list
		pass
	elif fill_values == None:
		fill_values = [None for _ in range(ver)]
	else:
		fill_values = [0 for _ in range(ver)]

	new = np.zeros([ver, hor])
	for v in range(ver):
		n_steps = shifts[v]
		if n_steps > 0:
			new[v:v+1, 0:n_steps] = fill_values[v]
			new[v:v+1, n_steps:] = mtrx[v:v+1, 0:hor-n_steps]
		elif n_steps < 0:
			n_steps = int(np.sqrt(n_steps**2)) # |ABS|
			new[v:v+1, 0:hor-n_steps] = mtrx[v:v+1, n_steps:hor]
			new[v:v+1, hor-n_steps:hor] = fill_values[v]
		else:
			new[v] = mtrx[v]
	return new



##################################################################################····3
def debt_matrix(inversiones, retrasos, times, amounts, length):
	N = len(inversiones)
	mx = np.zeros((N, length))
	
	for n in range(N):
		#print("times, monies", times, amounts)
		if times[n] == 0 or amounts[n] == 0:
			amort = 0
		else:
			amort = amounts[n]/times[n] # This should be its own variable: 'amort'
		months = times[n]
		ranger = np.arange(months, 0, -1)
		line = ranger[:] * amort
		mx[n:n+1,0:months] = line

	for n in range(N):
		retraso = retrasos[n]
		months = times[n]
		if retraso > 0:
			mx[n:n+1,retraso:length] = mx[n:n+1,0:length-retraso]
			mx[n:n+1,:retraso-1] = None # Use 0 to see graph
			mx[n:n+1,retraso-1:retraso] = 0 # Use 0 to see graph
		mx[n:n+1,months+retraso:] = None # Use 0 to see graph
	return mx

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

def smatrix(s, y_s): # GOOD. Info encapsulaiton.
	s = __shift_down__(s, y_s)
	zeros = [0 for _ in range(s.shape[0])]
	x_s = __x_intercepts_for_y__(s, targets=zeros)
	s = __cut_after__(s, targets=zeros)
	return s, x_s

def pre_debt(mmatrix, desembolso_matrix, debt_matrix, inversiones, caps, for_steps):
	N = debt_matrix.shape[0]
	#st.write(mmatrix[0:4,0] - inversiones + desembolso_matrix[0:4,0] + caps)
	#st.write(mmatrix[1:2,:for_steps[1]])
	#st.write(desembolso_matrix[1:2,:for_steps[1]])
	for n in range(N):
		#st.write(for_steps[n])
		m = n+1
		k = for_steps[n]
		debt_matrix[n:m,:k] = inversiones[n] - desembolso_matrix[n:m,:k] - caps[n]
	return debt_matrix

def keep_lowest_of_parallel_lists(a, b):
	for i in range(len(a)):
		a[i] = a[i] if a[i] <= b[i] else b[i]
	return a

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

def curva_matrix_old(curva_de_aplanamiento, N): # Returns N x 120 matrix UNFINISHED
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

def curva_steps2(N, min_, valorizaciones, length, periods=120):
	curvas = np.ones([N, length])
	for i in range(N):
		r = 1 + valorizaciones[i]/100
		r = np.e**(np.log(r)/12)
		curva_reducer = np.e**(np.log(min_/r)/periods)
		curvas[i:i+1,0:periods] = r * curva_reducer ** np.arange(1.0, periods+1)
		curvas[i:i+1,periods:] = min_
	return curvas

def duo_matrix(inputs, N, NN, income_matrix, D_matrix, y_s, MAX_LENGTH):
	zeros = [0 for _ in NN]

	rmatrix0 = income_matrix[:] + D_matrix[::] # + desembolso_matrix[:]
	rmatrix1 = income_matrix[:] + D_matrix[::] # + desembolso_matrix[:]
	y_r = [inputs[n]['cap_ini'] for n in NN]
	x_r = __x_intercepts_for_y__(rmatrix1, targets=y_r)

	duo_matrix = np.zeros([N*2, MAX_LENGTH])
	duo_matrix[0:N] = rmatrix0
	duo_matrix[N:N*2] = rmatrix1
	steps = y_s.copy()
	#print("Steps", steps)
	steps.extend(y_s)
	#print("Steps: ", steps); exit()
	duo_matrix = __shift_down__(duo_matrix, steps)
	y_duo = [inputs[n]['cap_ini'] for n in NN]
	y_duo.extend(y_duo)
	x_duo = __x_intercepts_for_y__(duo_matrix, targets=y_duo)
	duo_matrix[0:N] = __cut_after__(duo_matrix[0:N], targets=zeros)
	duo_matrix[N:N*2] = __cut_before__(duo_matrix[N:N*2], targets=zeros)
	duo_matrix[N:N*2] = __cut_after__(duo_matrix[N:N*2], targets=y_duo)
	return duo_matrix, x_duo, y_duo

def duo_matrix(mx, limits, end_values, maxlength):
	N = mx.shape[0]
	zeros = [0 for _ in range(N)]

	m_below_zero = mx.copy()
	m_over_zero  = mx.copy()
	
	y_r = limits
	x_r = __x_intercepts_for_y__(m_below_zero, targets=limits)

	duo_matrix = np.zeros([N*2, MAX_LENGTH])
	duo_matrix[0:N] = m_below_zero
	duo_matrix[N:N*2] = m_over_zero
	steps = [end_values[_] for _ in range(N)]
	steps.extend(end_values)
	duo_matrix = __shift_down__(duo_matrix, steps)
	#
	y_duo = limits
	y_duo.extend(y_duo)
	x_duo = __x_intercepts_for_y__(duo_matrix, targets=y_duo)
	duo_matrix[0:N] = __cut_after__(duo_matrix[0:N], targets=zeros)
	duo_matrix[N:N*2] = __cut_before__(duo_matrix[N:N*2], targets=zeros)
	duo_matrix[N:N*2] = __cut_after__(duo_matrix[N:N*2], targets=y_duo)
	return duo_matrix, x_duo, y_duo

def opor_sequence(inputs, r, length, untils):
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

def opor_sequence2(inputs, r, length, until=120): # Returns according to each desembolso, but─NOTE─does not shift by starting points.
	# a * (1-r**N) / (1-r)
	N = len(inputs)
	mx = np.zeros((N, length))
	ranger = np.zeros((N, length))
	ds = np.array([inputs[n]['max_desembolso_mensual'] for n in range(N)]).reshape(N, 1)
	#rs = np.array([inputs[n]['r_mes'] for n in range(N)]).reshape(N, 1)
	rs = np.array([r for _ in range(N)]).reshape(N, 1)
	invs = np.array([inputs[n]['cap_ini'] for n in range(N)]).reshape(N, 1)
	#us = np.array([u for u in range(N)]).reshape(N, 1)
	
	ranger[:] = np.arange(1, length+1)
	v = ranger[:,0:until].view()
	mx[:,0:until] = invs*rs**v + ds*(1-rs**v)/(1-rs)

	v = ranger[:,0:length-until].view()
	a = np.array(mx[:,until-1]).reshape(N, 1)
	mx[:,until:] = a*rs**v
	return mx

def opor_sequence3(cap_ini, opor_saving, r, length, until=120): # Returns according to each desembolso, but─NOTE─does not shift by starting points.
	# a * (1-r**N) / (1-r)
	N = 1
	mx = np.zeros((N, length))
	ranger = np.zeros((N, length))
	#ds = np.array([opor_saving for n in range(N)]).reshape(N, 1)
	#rs = np.array([inputs[n]['r_mes'] for n in range(N)]).reshape(N, 1)
	#invs = np.array([cap_ini for n in range(N)]).reshape(N, 1)
	ds = opor_saving
	rs = r
	invs = cap_ini
	
	ranger[:] = np.arange(1, length+1)
	v = ranger[:,0:until].view()
	mx[:,0:until] = invs*rs**v + ds*(1-rs**v)/(1-rs)

	v = ranger[:,0:length-until].view()
	a = np.array(mx[:,until-1]).reshape(N, 1)
	mx[:,until:] = a*rs**v
	return mx

def opor_sequence3(cap_ini, opor_saving, r, length, until=120): # Returns according to each desembolso, but─NOTE─does not shift by starting points.
	# a * (1-r**N) / (1-r)
	N = 1
	mx = np.zeros((N, length))
	ranger = np.zeros((N, length))
	#ds = np.array([opor_saving for n in range(N)]).reshape(N, 1)
	#rs = np.array([inputs[n]['r_mes'] for n in range(N)]).reshape(N, 1)
	#invs = np.array([cap_ini for n in range(N)]).reshape(N, 1)
	ds = opor_saving
	rs = r
	invs = cap_ini
	#print("rs, ", rs)
	ranger[:] = np.arange(1, length+1)
	v = ranger[:,0:until].view()
	mx[:,0:until] = invs*rs**v + ds*(1-rs**v)/(1-rs)

	v = ranger[:,0:length-until].view()
	a = np.array(mx[:,until-1]).reshape(N, 1)
	mx[:,until:] = a*rs**v
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

def shift_right(mx, list_): # .......?............... No work. elegantly.
	N = mx.shape[0]
	w = mx.shape[1]
	shifts = np.array(list_).reshape(N, 1)

	mx[::] = mx[:,0:shifts[:]] * None + mx[:,0:w-shifts[:]]
	return mx

def __shift_down__(*a, **k):
	return shift_down(*a, **k)

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

def __x_intercepts_for_y__(*a, **k):
	return x_intercepts_for_y(*a, **k)

def x_intercepts_for_y(mtrx, targets): # WORKS
	xs = {}
	ver = mtrx.shape[0]
	hor = mtrx.shape[1]

	# Rounds target number downwards.
	targets = [int(targets[number]) for number in range(len(targets))]
	
	for v in range(ver):
		for h in range(hor):
			#st.write(f'{v}:{h}')
			if mtrx[v][h] >= targets[v]:# and math.isnan(mtrx[v][h]) == True:
				xs[v] = h
			#	st.write("##########3··················································")
				break
	return xs

def to_dict(list_or_array, int_=False): # works
	d = {}
	for e in range(len(list_or_array)):
		#st.write("list_or_array", list_or_array)
		if int_==True:
			d.update({ e: int(list_or_array[e]) })
		else:
			d.update({ e: list_or_array[e] })
	return d

#def to_list

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

def __cut_after__(*a, **k):
	return cut_after(*a, **k)

def cut_after(mtrx, targets, start_from_end=False) -> 'mtrx': # WORKS
	ver = mtrx.shape[0]; hor = mtrx.shape[1]
	for v in range(ver):
		for h in range(hor):
			if mtrx[v][h] >= targets[v]:
				mtrx[v][h:] = None
				#mtrx[v][h] = None
				break
	return mtrx

def set_value_after_pos(mtrx, positions, value=0) -> 'mtrx':
	ver = mtrx.shape[0]
	for v in range(ver):
		mtrx[v][positions[v]:] = value
	return mtrx

def set_value_before_pos(mtrx, positions, value=0) -> 'mtrx':
	ver = mtrx.shape[0]
	for v in range(ver):
		mtrx[v][:positions[v]] = value
	return mtrx

def cut_after_x_pos(mtrx, positions) -> 'mtrx':
	ver = mtrx.shape[0]
	for v in range(ver):
		mtrx[v][positions[v]:] = None
	return mtrx

def set_None_after_x_pos(mtrx, positions) -> 'mtrx':
	ver = mtrx.shape[0]
	for v in range(ver):
		mtrx[v][positions[v]:] = None
	return mtrx

def set_to_X_values_greater_equal_Y(mtrx, y_values, fill=None):
	for n in range(mtrx.shape[0]):
		mtrx[n] = np.array(list(map(lambda i: i if i<=y_values[n] else fill, mtrx[n])))
	return mtrx

def cut_tail(mtrx, limits, fill=None):
	for n in range(mtrx.shape[0]):
		mtrx[n] = np.array(list(map(lambda i: i if float(i)>=limits[n] else fill, mtrx[n])))
	return mtrx

def repeat_last_after(mtrx, targets) -> 'mtrx': # WORKS
	ver = mtrx.shape[0]; hor = mtrx.shape[1]
	for v in range(ver):
		for h in range(hor):
			#st.write(f'v {v} : h {h} :: {mtrx[v][h]} >= {targets[v]}')
			if mtrx[v][h] >= targets[v]:
				mtrx[v][h:] = targets[v]
				break
	return mtrx

def repeat_last_after_X(mtrx, xs) -> 'mtrx': # WORKS
	V = mtrx.shape[0]#; H = mtrx.shape[1]
	for v in range(V):
		mtrx[v:v+1,xs[v]:] = mtrx[v:v+1,xs[v]]
	return mtrx

def __cut_before__(*a, **k):
	return cut_before(*a, **k)

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

def loans(inputs, processed, chain=False):
	N = len(inputs)
	if chain == True:
		return [inputs[n]['inversion'] - inputs[n]['cap_ini'] - inputs[n]['retraso'] * processed[n]['max_desembolso_mensual'] for n in range(N)]
	else:
		return [inputs[n]['inversion'] - inputs[n]['cap_ini'] - inputs[n]['retraso'] * inputs[n]['max_desembolso_mensual'] for n in range(N)]

def prices_of_loans(inputs, processed, chain=False):
	N = len(inputs)
	if chain == True:
		cash_prior = [inputs[n]['retraso'] * processed[n]['max_desembolso_mensual'] for n in range(N)]
	else:
		cash_prior = [inputs[n]['retraso'] * inputs[n]['max_desembolso_mensual'] for n in range(N)]
	principals = [-cash_prior[n] + inputs[n]['inversion']-inputs[n]['cap_ini'] for n in range(N)] 
	return [processed[n]['loan_repay_amount'] - principals[n] for n in range(N)]

def cash_spent_to_repay(inputs, processed): # No retraso or move...
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

def roi_matrix(inputs, mmatrix, desembolso_matrix, loan_matrix): # point - loan matrix / desembolso
	# value - loan at time - paid at time / (loan at time + paid)
	# (gain + desem) / desem
	N = len(inputs)
	invs = np.array([inputs[n]['inversion'] for n in range(N)]).reshape(N, 1)
	caps = np.array([inputs[n]['cap_ini'] for n in range(N)]).reshape(N, 1)
	mx = np.zeros([mmatrix.shape[0], mmatrix.shape[1]])
	#st.write("loan_matrix shape: ", loan_matrix.shape[1])

	for n in range(N):
		#mx[n:n+1,0:inputs[n]['retraso']] = (mmatrix[n:n+1,0:inputs[n]['retraso']] - invs[n] + desembolso_matrix[n:n+1,0:inputs[n]['retraso']] + caps[n]) / (desembolso_matrix[n:n+1,0:inputs[n]['retraso']] + caps[n])
		i = inputs[n]['inversion']
		k = inputs[n]['retraso']
		# To do: Set stop at payouts when loan is repayed.
		if k > 0: 
			# First, let's find ROI with theoretic static loan (before actual loan takes place, i.e., the diff between inv and own total cap invested).
			mx[n:n+1,0:k] = (mmatrix[n:n+1,0:k] - ((i - desembolso_matrix[n:n+1,0:k]) - caps[n])) / (desembolso_matrix[n:n+1,0:k] + caps[n]) # Imaginary loan
			# Then, let's find ROI once the loan starts running.
			mx[n:n+1,k:] = (mmatrix[n:n+1,k:] - loan_matrix[n:n+1,k:]) / (desembolso_matrix[n:n+1,k:] + caps[n])
		else:
			mx[n:n+1,:] = (mmatrix[n:n+1,:] - loan_matrix[n:n+1,:]) / (desembolso_matrix[n:n+1,:] + caps[n])
	return mx

def roi_matrix_minus_commission(com, inputs, growth_matrix, mmatrix, desembolso_matrix, loan_matrix):
	#reducer = (1-com)
	N = len(inputs)
	invs = np.array([inputs[n]['inversion'] for n in range(N)]).reshape(N, 1)
	caps = np.array([inputs[n]['cap_ini'] for n in range(N)]).reshape(N, 1)
	mx = np.zeros([mmatrix.shape[0], mmatrix.shape[1]])

	for n in range(N):
		#mx[n:n+1,0:inputs[n]['retraso']] = (mmatrix[n:n+1,0:inputs[n]['retraso']] - invs[n] + desembolso_matrix[n:n+1,0:inputs[n]['retraso']] + caps[n]) / (desembolso_matrix[n:n+1,0:inputs[n]['retraso']] + caps[n])
		i = inputs[n]['inversion']
		k = inputs[n]['retraso']
		# To do: Set stop at payouts when loan is repayed.
		if k > 0: 
			# First, let's find ROI with theoretic static loan (before actual loan takes place, i.e., the diff between inv and own total cap invested).
			mx[n:n+1,0:k] = (mmatrix[n:n+1,0:k] - growth_matrix[n:n+1,0:k]*com - ((i - desembolso_matrix[n:n+1,0:k]) - caps[n])) / (desembolso_matrix[n:n+1,0:k] + caps[n]) # Imaginary loan
			# Then, let's find ROI once the loan starts running.
			mx[n:n+1,k:] = (mmatrix[n:n+1,k:] - growth_matrix[n:n+1,k:]*com - loan_matrix[n:n+1,k:]) / (desembolso_matrix[n:n+1,k:] + caps[n])
		else:
			mx[n:n+1,:] = (mmatrix[n:n+1,:] - growth_matrix[n:n+1,:]*com - loan_matrix[n:n+1,:]) / (desembolso_matrix[n:n+1,:] + caps[n])

	return mx


def roi_minus_commission(rate, g_mx, p_net, spent):
	mx = (p_net - (g_mx * rate)) / spent
	return mx

def roi_earliest_max(roimx):
	N = roimx.shape[0]
	W = roimx.shape[1]

	for n in range(N):
		c = 0
		x = None
		for i in range(1, W):
			if roimx[n][w-1] < roimx[n][w] and roimx[n][w] > roimx[n][w+1]:
				x = w

def optimal_max_x(roimx):
	eln_matrix = np.e**(np.log(roimx)/np.arange(1,MAX_LENGTH+1))
	x_points = 1 + eln_matrix[:,1:].argmax(axis=1)
	return x_points		

def y_values_for_x(mx, x):
	Y = []
	for i in range(mx.shape[0]):
		Y.append(mx[i][x[i]])
	return Y

def datasheet(names, inversiones, roimx, roi_earliest_com, roi_max_com, Principals, debts, caps, interests, desembolsos, retrasos, amorts_repay_times, amorts_totals_closest):
	N = roimx.shape[0]

	data = {}
	for n in  range(N):
		allcaps = caps[n] + desembolsos[n]*retrasos[n]# + desembolsos[n]
		#st.write(f'Name: {names[n]} — allcaps: {allcaps}'); print("allcaps: ", allcaps); st.write("debts[n]: ", debts[n]); st.write("Principals[n]: ", Principals[n])
		#cash_out = caps[n] + [amorts_totals_closest]
		d = {
			'': names[n],
			'Repay: months ($)': f'{amorts_repay_times[n]} ({math.ceil(amorts_totals_closest[n])})',
			'Debt max': math.ceil(debts[n]),
			#'Loan repayed (closest)': int(amorts_totals_closest[n]),
			'Cost of loan (i payed)': math.ceil(interests[n]),
			'? Loan ratio: real (nom)': f'{round(debts[n]/inversiones[n], 2)} ({round(allcaps/(Principals[n]),2)})' if allcaps != 0 else f'─',
			#'? Loan ratio: real (nom)': f'{round(allcaps/(debts[n]+allcaps), 2)} ({round(allcaps/(Principals[n]),2)})' if allcaps != 0 else f'─',
			'Loan-to-earnings': '?',
			'Optimum ROI (+com)': round(roi_max_com[n], 3),
			'Optimal ROI time (incl.com.)': roi_earliest_com[n], #### When the slope was at steepest.
			'Highest risk moment': 12+roimx[n:n+1,retrasos[n]+1:].argmin(), # axis=1
		}
		#data[n] = { 'id': n, 'data': d }
		data[n] = d
	return data


###############################################################################################################

def loan_rangerOLD(tasas, inversiones, caps, desembolsos, length, retrasos, chain=False, desembolsos_alt=None):
	N = len(inversiones)
	mx = np.zeros((N, length))

	mx[:] = np.arange(1, length+1)

	for n in range(N):
		i = tasas[n]
		inversion = inversiones[n]
		cap_ini = caps[n]
		desembolso = desembolsos[n]
		retraso = retrasos[n]

		P = inversion - (cap_ini + desembolso*retraso)
		P = 0 if P < 0 else P
		#print(f'{n} loan_ranger: i: {i}, retraso: {retraso}, P: {P}')

		if i == 0 or i == None or math.isnan(i) == True: # If no interest rate
			mx[n][:] = P # 0 # Not None ?
		else:
			i = tasas[n]/12 
			#retraso=inputs[n]['retraso']
			#print(f'= {((P*i*(1+i)**(mx[n][:])) / ((1+i)**(mx[n][:]-retraso) - 1)) * (mx[n][:]-retraso)}')
			retraso = 0
			#mx[n][:] = ((P*i*(1+i)**mx[n][:]) / ((1+i)**mx[n][:] - 1)) # Each step is the monthly amort if that step represented the months.
			mx[n][:] = ((P*i*(1+i)**(mx[n][:]-retraso)) / ((1+i)**(mx[n][:]-retraso) - 1)) * (mx[n][:]-retraso)
	return mx
