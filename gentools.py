# General tools, easy to repeat
import numpy as np


def list_to_1d_matrix(l):
	height = len(l)
	return np.array(l).reshape(height, 1)


def list_to_1d_flat(l):
	height = len(l)
	return np.array(l).reshape(1, height)


def filter_matrix_greater_than(mx, limits_list, in_place_list) -> 'matrix':
	for n in range(mx.shape[0]):
		m = mx[n]#.view()
		t = limits_list[n]
		m[m>t] = t
		#mx[n][mx[n] > limits_list[n]] = in_place_list[n]
	return mx


def to_dict(list_or_array, key_type=int, value_type=int, as_int=False):
	d = {}
	for e in range(len(list_or_array)):
		d.update({ key_type(e): value_type(list_or_array[e]) })
	#for e in range(len(list_or_array)):
	#	if as_int==True:
	#		d.update({ e: int(list_or_array[e]) })
	#	else:
	#		d.update({ e: list_or_array[e] })
	return d


def keep_lowest_of_parallel_lists(a, b):
	for i in range(len(a)):
		a[i] = a[i] if a[i] <= b[i] else b[i]
	return a

# Makes a list vertical, then adds each item to every item in (corresponding line in) matrix.
def add_list_to_elements_in_matrix(mtrx, addlist):
	addmtrx = np.array(addlist)
	addmtrx = addmtrx.reshape(len(addlist), 1)
	return mtrx[:] + addmtrx


def cut_after_longest(mtrx, extra=0):
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

# Check if works. Should be slow, because it makes list, then numpy array.
def set_to_X_values_greater_equal_Y(mtrx, y_values, fill=None):
	for n in range(mtrx.shape[0]):
		mtrx[n] = np.array(list(map(lambda i: i if i<=y_values[n] else fill, mtrx[n])))
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


def make_table(mtrx, colnames, switches, x_axis_label='X', y_axis_label='Y', N=None):
	if N == None:
		N = mtrx.shape[0]
	data = []
	line_data = mtrx[0]
	length = line_data.shape[0]
	for l in range(N):
		if switches[l] == True:
			pass
		else:
			line_data = mtrx[l]
			for i in range(line_data.shape[0]):
				d = { 'name': colnames[l], x_axis_label: i, y_axis_label: line_data[i] }
				data.append(d)
	return data # return pd.DataFrame(data)

