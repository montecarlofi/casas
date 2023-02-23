import pandas as pd; #import get_data

def read_data(N=None):
	d = pd.read_csv("save.csv")
	names = list(d['name'])

	N = len(names); print("N ", N)

	inversiones = list(d['inversion'])
	tasas = list(d['tasa'])
	#capitales = list(d['capital'])
	try:
		caps_ini = list(d['cap_ini'])
	except Exception as e:
		caps_ini = capitales.copy()

	try:
		r_highs = list(d['r_high'])
	except Exception as e:
		r_highs = [2 for _ in range(N)]
	
	try:
		shifts = list(d['shift'])
	except Exception as e:
		shifts = [0 for _ in range(N)]
	
	#porciones = list(d['porcion'])
	#principales = list(d['principal'])
	#anos_con_prestamos = list(d['anos_con_prestamo'])
	#meses_con_prestamos = list(d['meses_con_prestamo'])
	#amorts_mes = list(d['amort_mes'])
	#amorts_totales = list(d['amort_total'])
	#deudas_y_capitales = list(d['deuda_y_capital'])
	#costos_prestamos = list(d['costo_prestamo'])
	valorizaciones = list(d['valorizacion'])
	rs_meses = list(d['r_mes'])
	ingresos_pesimistas = list(d['ingreso_pesimista'])
	ingresos_optimistas = list(d['ingreso_optimista'])
	retrasos = list(d['retraso'])
	hide_graphs = list(d['hide_graph'])

	#derrames_hasta_pagar_prestamos = list(d['derrame_hasta_pagar_prestamo'])
	#derrames_totales = list(d['derrame_total'])
	try:
		max_desembolsos_mensual = list(d['max_desembolso_mensual'])
	except Exception as e:
		max_desembolsos_mensual = [0 for i in range(N)]

	inns = [0 for _ in range(len(names))]
	for i in range(len(inns)):
		inn_data = {
			'name': names[i],
			'inversion': inversiones[i],
			'tasa': tasas[i],
			#'capital': capitales[i],
			'cap_ini': caps_ini[i],
			#'porcion': porciones[i],
			#'principal': principales[i],
			'valorizacion': valorizaciones[i],
			'r_high': r_highs[i],
			'r_mes': rs_meses[i],
			'ingreso_pesimista': ingresos_pesimistas[i],
			'ingreso_optimista': ingresos_optimistas[i],
			'retraso': retrasos[i],
			'shift': shifts[i],
			'max_desembolso_mensual': max_desembolsos_mensual[i],
			'hide_graph': hide_graphs[i]

			#'anos_con_prestamo': anos_con_prestamos[i],
			#'meses_con_prestamo': meses_con_prestamos[i],
			#'amort_mes': amorts_mes[i],
			#'amort_total': amorts_totales[i],
			#'deuda_y_capital': deudas_y_capitales[i],
			#'costo_prestamo': costos_prestamos[i],
			#'derrame_hasta_pagar_prestamo': derrames_hasta_pagar_prestamos[i],
			#'derrame_total': derrames_totales[i]
		}
		inns[i] = inn_data
	return inns
