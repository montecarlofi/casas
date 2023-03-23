import streamlit as st
import numpy as np
from CONSTANTS import *
LANG = 'Spanish'
if LANG == 'Spanish':
	from LANG_SPA import *
elif LANG == 'English':
	from LANG_ENG import *

def get_columns(N):
	#Columns = []
	#a, a0, b, b0, c, c0, d = st.columns([2, 1, 2, 1, 2, 1, 2])
	#Columns.append(a); Columns.append(b); Columns.append(c); Columns.append(d); 
	#colGraph = 0
	#return Columns, colGraph
	# Works:
	#colGraph, cola00, cola, cola0, colb, colb0, colc, colc0, cold = st.columns([9, 1, 2, 1, 2, 1, 2, 1, 2])
	colGraph, cola, colb, colc, cold = st.columns([6, 1, 1, 1, 1], gap='small')
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

	inversion = st.number_input(MONTO, value=inn['inversion'], key=f'inversion_{name}', disabled=disabled)
	cap_propio = st.number_input(CAPITAL, 0.0, inversion, value=inn['cap_ini'], key=f'cap_propio_{name}', disabled=disabled)
	if cap_propio > inversion:
		cap_propio = inversion
	tasa = st.number_input(TASA, 0.0, 30.0, value=inn['tasa']*100, key=f'tasa_{name}', disabled=disabled) / 100
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
def input_cols(inn, chained=False):
	#inn = { 'name': name }
	#st.write(name)
	disabled = True if inn['hide_graph'] == True else False

	R_high = int(inn['r_high'])
	#R_high = st.slider("High valoriz.", -5, 15, R_high, 1, key=f"high_{inn['name']}", disabled=disabled)
	valorizacion = int(inn['valorizacion'])
	valorizacion = st.slider(VALORIZACION, -5, 15, valorizacion, 1, key=f"valorizacion_{inn['name']}", disabled=disabled)
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
	ingreso_pesimista = st.slider(INGRESO, 0, 8000, value, 100, key=key, disabled=disabled)
	ingreso_optimista = 0#st.slider("Ingreso optimista K", 0, 5000, 2500, 100)
	ingreso_pesimista /= MONEY_MULTIPLIER
	ingreso_optimista /= MONEY_MULTIPLIER

	retraso = st.slider(PROYECTO_MESES, 0, 48, int(inn['retraso']), 6, key=f'retraso_{inn["name"]}', disabled=disabled)
	#shift = st.slider("Mover", 0, 96, int(inn['shift']), 1, key=f'shift_{inn["name"]}', disabled=disabled)
	if chained == False and disabled==True:
		true_false = True
	elif chained == True and disabled==False:
		true_false = True
	else:
		true_false = False
	if chained == True:
		true_false = True
	shift = st.number_input(MOVER, 0, 940, int(inn['shift']), key=f'shift_{inn["name"]}', disabled=true_false)

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
