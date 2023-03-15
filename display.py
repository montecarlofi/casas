import pandas as pd
import plotly.express as px
import streamlit as st

def streamlit_hide(markdown):
	hide_streamlit_style = """
	            <style>
	            #MainMenu {visibility: hidden;}
	            footer {visibility: hidden;}
	            </style>
	            """
	markdown(hide_streamlit_style, unsafe_allow_html=True) 

	markdown("""
	<style>
	div[data-testid="metric-container"] {
	   background-color: rgba(240, 242, 246, 0.7);
	   border: 1px solid rgba(240, 242, 246, 0.7);
	   padding: 10% 10% 10% 10%;
	   border-radius: 5px;
	   color: rgb(30, 103, 119);
	   overflow-wrap: break-word;
	}

	/* breakline for metric text         */
	div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
	   overflow-wrap: break-word;
	   white-space: break-spaces;
	   color: black;
	}
	</style>
	"""
	, unsafe_allow_html=True)

	#st.metric(label="This is a very very very very very long sentence", value="70 °F")

def plot(inputs, mtrx, xs, ys, colours=None, show_labels=True, labels=None):
	#try:
	#	if labels == None:
	#		labels = xs		
	#except Exception as e:
	#	pass
	#print("labels ", labels); exit()
	#for i in range(mtrx.shape[0]):
	#	labels[i] = int(labels[i])
		#print(f"Label {i} is {labels[i]}")

	N = mtrx.shape[0]

	data = []
	for l in range(N):
		line_data = mtrx[l]
		for i in range(line_data.shape[0]):
			d = { 'name': inputs[l]['name'], 'mes': i, 'y': line_data[i], 'loan_repay_time': 1 }
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
		#print(key, ys[key])
		#if MAX_LENGTH >= x_intercepts[i]:
		if key == 0 or key == 1:
			yshift=8
		else:
			yshift=8
		#print("Key: ", key)
		if inputs[key]['hide_graph'] == False or show_labels == False:
			if xs[key] <= mtrx.shape[1]:
				if inputs[key]['shift'] == 0:
					text = labels[key]
				else:
					text = f'{labels[key]-inputs[key]["shift"]} ({labels[key]})'
				plot.add_annotation(x=xs[key], y=ys[key],#y_intercepts[key],
					text=text,
					showarrow=False,
					yshift=yshift)

	plot.update_yaxes(visible=True, showticklabels=True)
	plot.update_layout(showlegend=True)
	st.plotly_chart(plot, use_container_width=True)

def plot2(mtrx, xs, ys, N, names=None, hides=None, colours=None, show_labels=True, labels=None, message=""):
	N = mtrx.shape[0]
	#st.write("Message: ", message)
	#st.write	(mtrx.shape[0], " : ", mtrx.shape[1])
	#st.write("Names", names)
	if names == None:
		names = [i for i in range(N)]
	if hides == None:
		hides = [False for i in range(N)]

	data = []
	for l in range(N):
		line_data = mtrx[l]
		for i in range(line_data.shape[0]):
			#print("name: ", names[l])
			d = { 'name': names[l], 'mes': i, 'y': line_data[i] }
			#if inputs[l]['ingreso_pesimista'] == 0:
			if hides[l] == True:
				pass
			else:
				data.append(d)
				#print(d)

	df = pd.DataFrame(data)

	if colours == None:
		colours = ['green', 'blue', 'orange', 'red', 'lightgreen', 'lightblue', 'rgb(254, 217, 166)', 'pink', 'lightgrey']
	colourmap = {}
	for c in range(len(names)):
		colourmap.update({
			names[c]: colours[c]
		})

	plot = px.line(df, x=df.mes, y=df.y, hover_name=df.name, color='name',  
		color_discrete_map=colourmap
		)

	for key in xs:
		#print(key, ys[key])
		#if MAX_LENGTH >= x_intercepts[i]:
		if key == 0 or key == 1:
			yshift=8
		else:
			yshift=8
		#print("Key: ", key)
		if hides[key] == False or show_labels == False:
			if xs[key] <= mtrx.shape[1]:
				plot.add_annotation(x=xs[key], y=ys[key],#y_intercepts[key],
					text=(f'{labels[key]}'),
					showarrow=False,
					yshift=yshift)

	plot.update_yaxes(visible=True, showticklabels=True)
	plot.update_layout(showlegend=True)
	st.plotly_chart(plot, use_container_width=True)

def plot_duo(inputs, mtrx, xs, ys, colours=None, show_labels=True, labels=None, message=None):
	names = [inputs[n]['name'] for n in range(len(inputs))]
	names.extend(names)
	for i in range(len(inputs), len(names)):
		names[i] = names[i] + str('b')
	hide_graphs = [inputs[n]['hide_graph'] for n in range(len(inputs))]
	hide_graphs.extend(hide_graphs)
	#print(hide_graphs); exit()
	N = mtrx.shape[0]
	#print(hide_graphs)

	data = []
	for l in range(N):
		line_data = mtrx[l]
		for i in range(line_data.shape[0]):
			d = { 'name': names[l], 'mes': i, 'y': line_data[i], 'blob': 1 }
			#if inputs[l]['ingreso_pesimista'] == 0:
			if hide_graphs[l] == True:
				pass
			else:
				data.append(d)

	df = pd.DataFrame(data)

	if colours == None:
		colours = ['green', 'blue', 'orange', 'red', 'lightgreen', 'lightblue', 'rgb(254, 217, 166)', 'pink']
	colourmap = {}
	for c in range(N):
		colourmap.update({
			f'{names[c]}': colours[c]
		})
	#print("Colourmap: ", colourmap)
	plot = px.line(df, x=df.mes, y=df.y, hover_name=df.name, color='name',  
		color_discrete_map=colourmap
		)
	st.write("Message: ", message); print("Message: ", message); 
	#for key in xs:
	#st.write(xs)
	#for key in range(len(xs)):
	for key in range(len(inputs)):
		#st.write(key, ys[key])
		#if MAX_LENGTH >= x_intercepts[i]:
		if key == 0 or key == 1:
			yshift=8
		else:
			yshift=8
		#print("Key: ", key)

		if hide_graphs[key] == False or show_labels == False:
			print("key: ", key)
			if xs[key] <= mtrx.shape[1]:
				if inputs[key]['shift'] == 0:
					text = labels[key]
				else:
					text = f'{labels[key]-inputs[key]["shift"]} ({labels[key]})'
				plot.add_annotation(x=xs[key], y=ys[key],#y_intercepts[key],
					#text=(f'{labels[key]}'),
					text=text,
					showarrow=False,
					yshift=yshift)

	plot.update_yaxes(visible=True, showticklabels=True)
	plot.update_layout(showlegend=True)
	st.plotly_chart(plot, use_container_width=True)

