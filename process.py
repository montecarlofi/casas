import math
import numpy as np # import pandas as pd
from CONSTANTS import * 
import streamlit as st


# Curva de valorización à la colombiana.
def curva_steps(N, min_, valorizaciones, length, periods=120):
	curvas = np.ones([N, periods])
	curvas_next = np.ones([N, 1]) * min_
	for i in range(N):
		#r = 1 + valorizaciones[i]/100
		#r = np.e**(np.log(r)/12)
		r = valorizaciones[i]
		curva_reducer = np.e**(np.log(min_/r)/periods)
		curvas[i:i+1,0:periods] = r * curva_reducer ** np.arange(1.0, periods+1)
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


# Exponential growth matrix (cumulative value in steps).
def growth_matrix(inversiones, rates, length, curva_flag=False, curvas=None, curva_next=0):
	N = len(inversiones)
	invs = np.array(inversiones).reshape(N, 1)
	
	mx = np.zeros((N, length))

	if curva_flag == False:
		rs = np.array(rates).reshape(N, 1)
		mx[::] = np.arange(1, length+1) # Exponents.
		mx[::] = invs*rs**mx

	else:
		length_ = curvas.shape[1]
		mx[:,0:1] = invs[:]*curvas[:,0:1]
		for i in range(1, length_):
			mx[:,i:i+1] = mx[:,i-1:i] * curvas[:,i:i+1]
		mx[:,length_:] = np.arange(1, length-length_+1) # Exponents.
		mx[:,length_:] = mx[:,length_-1:length_] * curva_next ** mx[:,length_:]

	return mx


# Loan size range, calculated by repayment times (x-axis).
def loan_ranger(Ps, tasas, length):
	N = len(Ps)
	mx = np.zeros((N, length))

	mx[:] = np.arange(1, length+1)

	for n in range(N):
		i = tasas[n]
		P = Ps[n]
		if i == 0 or i == None or math.isnan(i) == True: # If no interest rate.
			mx[n][:] = P
		else:
			i = tasas[n]/12 
			mx[n][:] = (P*i*(1+i)**(mx[n][:]) / ((1+i)**(mx[n][:]) - 1)) * (mx[n][:])
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
		#ab, a_index, b_index = np.intersect1d(a, b, return_indices=True)
		#xs.append(b_index)
		#print("index:", a_index)
#		index_of_first = np.argmax((a <= b) == True)
		#print("index: ", index_of_first)
		#xs.append(index_of_first)
		#ys.append(b[index_of_first])
	return xs, ys


def get_and_filter_principals(inversiones, caps, retrasos, desembolsos, ingresos):
	ps = [inversiones[n] - caps[n] - retrasos[n]*desembolsos[n] for n in range(len(inversiones))]
	return list(map(lambda x: x if x >= 0 else 0, ps))


def shift_sequences(mtrx, shifts, fill_values=None): # Positive shifts right; negative, left.
	ver = mtrx.shape[0]
	hor = mtrx.shape[1]
	
	if hasattr(fill_values, "__len__") == True: # Is list?
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


def smatrix(s, y_s):
	s = __shift_down__(s, y_s)
	zeros = [0 for _ in range(s.shape[0])]
	x_s = __x_intercepts_for_y__(s, targets=zeros)
	s = __cut_after__(s, targets=zeros)
	return s, x_s


def pmatrix2(growth_matrix, income_matrix, spent, inversiones, amorts_totals_closest, amorts_repay_times, retrasos, shifts, out_of_bounds_value=None):
	N = len(inversiones)
	pmatrix = growth_matrix - inversiones + income_matrix - amorts_totals_closest
	y_p = [spent[n][amorts_repay_times[n]+retrasos[n]] for n in range(N)] # Todo el dinero gastado
	x_p, y = __x_intercepts_for_y__(pmatrix, targets=y_p, out_of_bounds_value=out_of_bounds_value)
	x_p = __to_dict__([int(x_p[n])+shifts[n] for n in range(N)])
	return pmatrix, x_p, y_p


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

def duo_matrix_new(mx, limits, end_values, maxlength):
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


def opor_sequence3(cap_ini, opor_saving, r, length, until=120): # Returns according to each desembolso, but─NOTE─does not shift by starting points.
	N = 1
	mx = np.zeros((N, length))
	ranger = np.zeros((N, length))
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


def shift_right(mx, list_): # .......?............... No work. elegantly.
	N = mx.shape[0]
	w = mx.shape[1]
	shifts = np.array(list_).reshape(N, 1)

	mx[::] = mx[:,0:shifts[:]] * None + mx[:,0:w-shifts[:]]
	return mx


def shift_down(mx, y_values): # WORKS
	y_values = np.array(y_values).reshape(mx.shape[0], 1)
	#for i in range(mx.shape[1]):
	#	mx[:,i] = mx[:,i] - y_values
	return mx[:] - y_values

def __shift_down__(*a, **k):
	return shift_down(*a, **k)


def x_intercepts_for_y(mtrx, targets, out_of_bounds_value=-1):
	xs = {}
	ys = {}
	ver = mtrx.shape[0]
	hor = mtrx.shape[1]

	# Rounds target number downwards.
	targets = [int(targets[number]) for number in range(len(targets))]
	
	for v in range(ver):
		xs[v] = out_of_bounds_value
	
	for v in range(ver):
		for h in range(hor):
			#st.write(f'{v}:{h}')
			if mtrx[v][h] >= targets[v]:# and math.isnan(mtrx[v][h]) == True:
				xs[v] = h
				ys[v] = mtrx[v][h]
			#	st.write("##########3··················································")
				break
	return xs, ys # Remember: ys values can be the same or above targets.

def __x_intercepts_for_y__(*a, **k):
	return x_intercepts_for_y(*a, **k)


def to_dict(list_or_array, int_=False): # works
	d = {}
	for e in range(len(list_or_array)):
		#st.write("list_or_array", list_or_array)
		if int_==True:
			d.update({ e: int(list_or_array[e]) })
		else:
			d.update({ e: list_or_array[e] })
	return d

def __to_dict__(*a, **k):
	return to_dict(*a, **k)


def len_longest_graph(mtrx):
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


def cut_after(mtrx, targets, start_from_end=False) -> 'mtrx': # WORKS
	ver = mtrx.shape[0]; hor = mtrx.shape[1]
	for v in range(ver):
		for h in range(hor):
			if mtrx[v][h] >= targets[v]:
				mtrx[v][h:] = None
				#mtrx[v][h] = None
				break
	return mtrx

def __cut_after__(*a, **k):
	return cut_after(*a, **k)


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


def cut_tail(mtrx, limits, fill=None):
	for n in range(mtrx.shape[0]):
		mtrx[n] = np.array(list(map(lambda i: i if float(i)>=limits[n] else fill, mtrx[n])))
	return mtrx


def roi_minus_commission(rate, g_mx, p_net, spent):
	mx = (p_net - (g_mx * rate)) / spent
	return mx


def optimal_max_x(roimx, start_point=0):
	eln_matrix = np.e**(np.log(roimx)/np.arange(1,MAX_LENGTH+1))
	x_points = start_point + eln_matrix[:,start_point:].argmax(axis=1)
	return x_points


def y_values_for_x(mx, x):
	Y = []
	for i in range(mx.shape[0]):
		Y.append(mx[i][x[i]])
	return Y


def datasheet(names, inversiones, roimx, roi_max, roi_earliest, roi_earliest_com, roi_max_com, gains_roiearl_com, gains_at_entrega, Principals, debts, caps, interests, desembolsos, retrasos, amorts_repay_times, amorts_totals_closest, cost_opt_roi_com_time):
	N = roimx.shape[0]

	data = {}
	for n in  range(N):
		allcaps = caps[n] + desembolsos[n]*retrasos[n]# + desembolsos[n]
		#st.write(f'Name: {names[n]} — allcaps: {allcaps}'); print("allcaps: ", allcaps); st.write("debts[n]: ", debts[n]); st.write("Principals[n]: ", Principals[n])
		#cash_out = caps[n] + [amorts_totals_closest]
		d = {
			'': names[n],
			'Saldar: N meses': f'{amorts_repay_times[n]}',
			'Deuda max, $': f'{math.ceil(debts[n])} ({math.ceil(amorts_totals_closest[n])})',
			'Costo préstamo (ints.) [hold] $': math.ceil(interests[n]),
			'Costo @opt ROI-tiempo [sell] $': int(cost_opt_roi_com_time[n]),
			#'Debt-to-principal ratio': round(allcaps/Principals[n] if Principals[n] != 0 else 1,2),
			#'Loan repayed (closest)': int(amorts_totals_closest[n]),
			#'Debt-to-investment ratio': round(debts[n]/inversiones[n], 2),
			#'? Loan ratio: real (nom)': f'{round(debts[n]/inversiones[n], 2)} ({round(allcaps/(Principals[n]),2)})' if allcaps != 0 else f'─',
			#'? Loan ratio: real (nom)': f'{round(allcaps/(debts[n]+allcaps), 2)} ({round(allcaps/(Principals[n]),2)})' if allcaps != 0 else f'─',
			#'Debt-to-earnings': '',
			#'Cap-a-inv cociente': str(int(100*allcaps/inversiones[n] if inversiones[n] != 0 else 1)) + "%",
			'Cap-a-inv cociente': round(allcaps/inversiones[n], 2) if inversiones[n] != 0 else 1,
			#'Optimum ROI': round(roi_max[n], 3),
			#'Optimal ROI time': roi_earliest[n], #### When the slope was at steepest.
			'Neta gan. entrega proy., $': int(gains_at_entrega[n]),
			'ROI-tiempo opt. (-com.), N': roi_earliest_com[n], #### When the slope was at steepest.
			'Neta opt ROI-tiempo, $': int(gains_roiearl_com[n]),
			'ROI óptimo (-comisiones)': str(int(100*(roi_max_com[n]-1))) + '%',
			#'Highest risk moment': retrasos[n]+roimx[n:n+1,retrasos[n]+1:].argmin(), # axis=1
		}
		#data[n] = { 'id': n, 'data': d }
		data[n] = d
	return data