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

def to_dict(list_or_array, as_int=False):
	d = {}
	for e in range(len(list_or_array)):
		if as_int==True:
			d.update({ e: int(list_or_array[e]) })
		else:
			d.update({ e: list_or_array[e] })
	return d

