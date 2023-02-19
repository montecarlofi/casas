import pandas as pd
import plotly.express as px
import streamlit as st

def plot_llegar_a_0(inputs, x_intercepts, imatrix, N):
	data = []
	for l in range(N):
		line_data = imatrix[l]
		for i in range(line_data.shape[0]):
			d = { 'name': inputs[l]['name'], 'mes': i, 'y': line_data[i], 'blob': 1 }
			#if inputs[l]['ingreso_pesimista'] == 0:
			if inputs[l]['hide_graph'] == True:
				pass
			else:
				data.append(d)

	df = pd.DataFrame(data)#; print(df); exit()

	plot0 = px.line(df, x=df.mes, y=df.y, hover_name=df.name, color='name',  
		color_discrete_map={ inputs[0]['name']: 'green', inputs[1]['name']: 'blue', inputs[2]['name']: 'orange', inputs[3]['name']: 'red'}
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
			plot0.add_annotation(x=x_intercepts[key], y=0,#y_intercepts[key],
				text=(f'{x_intercepts[key]}'),
				showarrow=False,
				yshift=yshift)

	plot0.update_layout(showlegend=True)
	st.plotly_chart(plot0, use_container_width=False)

def plot(inputs, mtrx, xs, ys, colours=None):
	data = []
	for l in range(mtrx.shape[0]):
		line_data = mtrx[l]
		for i in range(line_data.shape[0]):
			d = { 'name': inputs[l]['name'], 'mes': i, 'y': line_data[i], 'blob': 1 }
			#if inputs[l]['ingreso_pesimista'] == 0:
			if inputs[l]['hide_graph'] == True:
				pass
			else:
				data.append(d)

	df = pd.DataFrame(data)

	if colours == None:
		colours = ['green', 'blue', 'orange', 'red', 'lightgreen', 'lightblue', 'rgb(254, 217, 166)', 'pink']
	colourmap = {}
	for c in range(len(inputs)):
		colourmap.update({
			inputs[c]['name']: colours[c]
		})

	plot = px.line(df, x=df.mes, y=df.y, hover_name=df.name, color='name',  
		color_discrete_map=colourmap
		)

	for key in xs:
#		print(key, ys[key])
		#if MAX_LENGTH >= x_intercepts[i]:
		if key == 0 or key == 1:
			yshift=8
		else:
			yshift=8
		if inputs[key]['hide_graph'] == False:
			if xs[key] <= mtrx.shape[1]:
				plot.add_annotation(x=xs[key], y=0,#y_intercepts[key],
					text=(f'{xs[key]}'),
					showarrow=False,
					yshift=yshift)

	plot.update_yaxes(visible=True, showticklabels=True)
	plot.update_layout(showlegend=True)
	st.plotly_chart(plot, use_container_width=False)