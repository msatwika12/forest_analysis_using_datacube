from flask import Flask,render_template, request,flash,jsonify,redirect
import psycopg2
import pandas as pd
import numpy as np
import datacube
from deafrica_tools.plotting import rgb, display_map
import datacube
import odc.algo
import matplotlib.pyplot as plt
from datacube.utils.cog import write_cog
from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.plotting import display_map, rgb
import io
import base64
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import leaflet.pm;
# import leaflet.pm/dist/leaflet.pm.css;
# global area_per_pixel


#display_map(x=lon_range, y=lat_range)


print("----------------------------------\n")
 
app = Flask(__name__)
app.secret_key='your_secret_key'

image_base64=''
total_area_km2=0
@app.route('/',methods=['POST','GET'])
def hello_world():
	
	# content_type = request.headers.get('Content-Type')
	# print(content_type," ----- --- -- - -- * - * - ")
#    (content_type == 'application/json'):
	

	return render_template("common.html")

	# return render_template('image.html',img_base64=image_base64)
@app.route('/about', methods=['POST','GET'])
def process():

	return render_template('about.html')
		
@app.route('/form',methods=['POST','GET'])
def form():
	image_base64=''
	total_area_km2=0
	total_area_array=[]
	pl_a=[]
	ml_a=[]
	ofm_array=[]
	dfm_array=[]
	sfm_array=[]

	# content_type = request.headers.get('Content-Type')
	# print(content_type," ----- --- -- - -- * - * - ")
#    (content_type == 'application/json'):
	if request.method=='POST' :
		print("--------------------------")
		st= request.form.get('start-date')
		en = request.form.get('end-date')
		op = request.form.get('option')
		x = request.form.get('x')
		y = request.form.get('y')
		print(st,en,op,x,y)
		
		x_ti= json.loads(x)
		y_ti = json.loads(y)
		
		print("------------------------------------------------------------------")
		print(x_ti)
		
		print(y_ti)
		
		print("--------------------------------------------------------------------------------")
		for i in range(len(x_ti)):
			print("--------  ------------------------------------------------------------------------*---")
			
			print(x_ti[i][0],y_ti[i][0])
			print(x_ti[i][1],y_ti[i][1])
			# print(i,x[i][0])
			# print(x[i],x[i][0])
			print("-----------------------------------------------------------------------------**---")
		
			lon_range = (x_ti[i][0],y_ti[i][0])
			lat_range = (x_ti[i][1],y_ti[i][1])
		# ------------------------------------
		# lat_range = (15.65, 15.95)
		# lon_range = (80.75, 81.05)
			print(lon_range)
			print(lat_range,"   iiiiiiiiiiiiiiii")
			time_range = (st,en)
			# display_map(x=lon_range, y=lat_range)
			dc = datacube.Datacube(app="04_Plotting")
			ds = dc.load(product="s2a_sen2cor_granule",
							measurements=["B04_10m","B03_10m", "B08_10m"],

						x=lon_range,
						y=lat_range,
						time=time_range,
						output_crs='EPSG:6933',
						resolution=(-30, 30))

			print(ds)
			if not ds:
				
				return jsonify({'message':" not null"})
		

# Get the spatial resolution of the dataset
			spatial_resolution = np.abs(ds.geobox.affine.a)

	# Calculate the area per pixel
			area_per_pixel = spatial_resolution**2

	# Determine the number of pixels in the dataset
			num_pixels = ds.sizes['x'] * ds.sizes['y']

	# Calculate the total area
			total_area = area_per_pixel * num_pixels


			total_area_km2=total_area/1000000
			total_area_array.append(total_area_km2)


			print("Area per pixel: {} square meters".format(area_per_pixel))
			print("Total area: {} square meters".format(total_area))
			print("Total area: {} square kms".format(total_area_km2))
			dataset =  odc.algo.to_f32(ds)
		
			# ds=calculate_indices(ds,op,satellite_mission='s2')
	
			if op=='NDVI':
				band_diff = dataset.B08_10m - dataset.B04_10m
				band_sum = dataset.B08_10m + dataset.B04_10m
				ds_index = band_diff/band_sum
				#ds=calculate_indices(ds,op,satellite_mission='s2')
				
				#ds_index = ds.NDVI
				plt.figure()
				ds_index.plot.hist(figsize=(9,4))
				plt.xlabel(' NDVI_ Value')
				plt.ylabel('NO_OF_PIXELS')
				plt.title('Histogram')
				buffer = io.BytesIO()
				plt.savefig(buffer, format='png')
				buffer.seek(0)
			

				image_base64_2=base64.b64encode(buffer.read()).decode('utf-8')
				buffer.close()

				print('ndvi')
			
			
			

		# print(ndvi)
		# Generate the plot
		# for i in range(len(ds_index)):
			# plt.figure()
			#ds=calculate_indices(ds,'NDVI',satellite_mission='s2')
			#ds_index=ds.NDVI
			band_diff = dataset.B08_10m - dataset.B04_10m
			band_sum = dataset.B08_10m + dataset.B04_10m
			ds_index = band_diff/band_sum
			
			dense_forest_mask = np.where((ds_index > 0.6) & (ds_index < 0.8), 1, 0)
			open_forest_mask = np.where((ds_index > 0.3) & (ds_index < 0.6) , 1, 0)
			x=np.where((ds_index>0.8) | (ds_index<0.1),1,0)

			sparse_forest_mask = np.where((ds_index > 0.1) & (ds_index < 0.3) , 1, 0)
			f_a=[]
			
			w=np.sum(x[0])
			print(np.sum(x[0]),"---",area_per_pixel)
			ta=area_per_pixel*w
			ta2=ta/1000000
			print(ta,ta2)
			time_values = ds_index.time.values
			d=['time','dfm','ofm','sfm','tfa']
			x=[]
			for i in range(len(dense_forest_mask)):
				w=[]
				w.append(pd.to_datetime(time_values[i]))
				w.append(area(dense_forest_mask[i],area_per_pixel))
				w.append(area(open_forest_mask[i],area_per_pixel))
				w.append(area(sparse_forest_mask[i],area_per_pixel))
				w.append(area(dense_forest_mask[i],area_per_pixel)+area(open_forest_mask[i],area_per_pixel)+area(sparse_forest_mask[i],area_per_pixel))
				f_a.append(area(dense_forest_mask[i],area_per_pixel)+area(open_forest_mask[i],area_per_pixel)+area(sparse_forest_mask[i],area_per_pixel))
				x.append(w)
	# df['time'] = pd.to_datetime(time_values)
			df = pd.DataFrame(x, columns=d)
			print(df)

# Assuming your time column is named 'time' and the value column is named 'ndvi'
# Convert the 'time' column to pandas timetime if it's not already in that format
# Read the CSV file into a pandas DataFrame

			df['time'] = pd.to_datetime(df['time'])
			print(df.head())
			X_train=df['time']
			y_train=df['tfa']
	# Split the data into training and test sets
	# X_train, X_test, y_train, y_test = train_test_split(df['time'], df['dfm'],test_size=0.3,shuffle=False)

	# Extract the time components as features
			X_train_features = pd.DataFrame()
			X_train_features['year'] = X_train.dt.year
			X_train_features['month'] = X_train.dt.month
			X_train_features['day'] = X_train.dt.day
			# Add more features as per your requirements

			# Initialize and fit the Random Forest Regressor model
			model = RandomForestRegressor()
			model.fit(X_train_features, y_train)

			# Extract features from the test data
			X_test_features = pd.DataFrame()
			X_test_features['year'] = [2018]
			X_test_features['month'] =[ 5]
			X_test_features['day'] = [5]
			# Add more features as per your requirements
			print(X_train_features.head())
			print(X_test_features.head())
			# Predict the values
			predictions = model.predict(X_test_features)

			# Print the predictions
			print(predictions)
			prede=model.predict(X_train_features)
			print(prede)
		# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		# y_train=df['dfm']
		# model = RandomForestRegressor()
		# model.fit(X_train_features, y_train)
		# prede1=model.predict(X_train_features)
		# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		# y_train=df['sfm']
		# model = RandomForestRegressor()
		# model.fit(X_train_features, y_train)
		# prede2=model.predict(X_train_features)
		# plt.figure()
		# ds_index.plot(col='time',vmin=-1,vmax=1)
		
		
		
		# # Convert the plot to a PNG image in memory
		# buffer1= io.BytesIO()
		# plt.savefig(buffer1, format='png')
		# buffer1.seek(0)
		# image_base64_1 =  base64.b64encode(buffer1.read()).decode('utf-8')
		# buffer1.close()
		# plt.figure()
		# ds_index[0].plot()
		# buffer2=io.BytesIO()
		# plt.savefig(buffer2,format='png')
		# buffer2.seek(0)
		# image_base64_2=base64.b64decode(buffer2.read()).decode('utf-8')
		# print("-------------------------------",image_base64_2)
			
			if op=='NDWI':
				band_diff = dataset.B08_10m - dataset.B04_10m
				band_sum = dataset.B08_10m + dataset.B04_10m
				ds_index = band_diff/band_sum
				plt.figure(figsize=(9,4))
				ds_index.plot.hist(figsize=(9,4))
				plt.xlabel('NDWI_Value')
				plt.ylabel('NO_OF_PIXELS')
				plt.title('Histogram')
				buffer = io.BytesIO()
				plt.savefig(buffer, format='png')
				buffer.seek(0)
			

				image_base64_2=base64.b64encode(buffer.read()).decode('utf-8')
				

				print('ndwi')
				buffer.close()
			plt.figure()
			print(len(ds['time']))
			if(len(ds['time'])==1):
				print("entered into one ")
				plt.figure()
				ds_index.plot(vmin=-1,vmax=1)
				
				buffer = io.BytesIO()
				plt.savefig(buffer, format='png')
				buffer.seek(0)
				image_base64_1 = base64.b64encode(buffer.read()).decode('utf-8')
				buffer.close()
			elif(len(ds['time'])>1):
				print("entered into multiple")
				# plt.figure()
				# ds_index.plot(col='time', vmin=-1, vmax=1)
				
				# buffer = io.BytesIO()
				# plt.savefig(buffer, format='png')
				# buffer.seek(0)
				# image_base64_1 = base64.b64encode(buffer.read()).decode('utf-8')
				# buffer.close()
				# Create a figure and subplots
				fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

				# Plot on the first subplot
				plot1 =ds_index[0].plot(ax=axes[0], vmin=-1, vmax=1, col_wrap=3)

				# Plot on the second subplot
				plot2 = ds_index[-1].plot(ax=axes[1], vmin=-1, vmax=1, col_wrap=3)
				
				axes[0].set_title(str(ds_index.time.values[0]).split('T')[0])
				axes[1].set_title(str(ds_index.time.values[-1]).split('T')[0])

			

				# Hide the axis of the third subplot
				#axes[2].axis('off')
				plt.subplots_adjust(wspace=0.3)
				
				buffer = io.BytesIO()
				plt.savefig(buffer, format='png')
				buffer.seek(0)
				image_base64_1 = base64.b64encode(buffer.read()).decode('utf-8')
				buffer.close()

			# # Convert the plot to a PNG image in memory
			# plt.figure()
			# buffer = io.BytesIO()
			# plt.savefig(buffer, format='png')
			# buffer.seek(0)
			# image_base64_1 = base64.b64encode(buffer.read()).decode('utf-8')
			# buffer.close()
			indices = np.arange(len(X_train))
			print("indices : : : ===  ",indices)

			# plt.figure()
			# ds_index[0].plot()

			# # Convert the plot to a PNG image in memory
			# buffer = io.BytesIO()
			# plt.savefig(buffer, format='png')
			# buffer.seek(0)
			# image_base64_2 = base64.b64encode(buffer.read()).decode('utf-8')
			# buffer.close()
			if op=='ML PREDICTION':


			
			# Plot the actual values
				df['date']=pd.to_datetime(df['time'])
				plt.figure()
				plt.plot(df['date'], df['tfa'], color='blue', label='Actual')
			# plt.plot(indices, df['dfm'], color='blue', label='Actual2')
			# plt.plot(indices, df['sfm'], color='blue', label='Actual3')
			# Plot the predicted values
				plt.plot(df['date'],prede, color='red', label='Predicted')
			# plt.plot(indices, prede1, color='green', label='Predicted2')
			# plt.plot(indices, prede2, color='yellow', label='Predicted3')
			# plt.plot(indices, total_area_km2, color='blue', label='actual ')
			# plt.plot(df['time'],df['tfa'], color='yellow', label='Predicted3')

				# Add labels and title
				plt.xlabel('Date')
				plt.ylabel('Forest_area_km2')
				plt.title(' Forest Predictions')

				# Add legend
				plt.legend()
				buffer=io.BytesIO()
				plt.savefig(buffer, format='png')

				buffer.seek(0)
				image_base64_2=base64.b64encode(buffer.read()).decode('utf-8')
				buffer.close()
				
		

				print('ndwi')
				buffer.close()
			
			if op=='FOREST AREA':
				plt.figure()
				df['date']=pd.to_datetime(df['time'])
				plt.plot(df['date'],f_a,marker='o')
				plt.xlabel('DATE')
				plt.ylabel('Forest_area_km2')
				plt.title('Forest cover')
				buffer=io.BytesIO()
				plt.savefig(buffer, format='png')

				buffer.seek(0)
				image_base64_2=base64.b64encode(buffer.read()).decode('utf-8')

				image_base_2=base64.b64encode(buffer.read()).decode('utf-8')
				buffer.close()
			ml_a.append(image_base64_2)
			pl_a.append(image_base64_1)
		# Display the graph
		# print("Image 1 Base64:", image_base64_1)
		# print("Image 2 Base64:", image_base64_2)
		# image_base64_2 = image_to_base64(ds_index)

		# # Encode the PNG image as a base64 string
		# image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
		
		# plt.close()
		# plt.figure()
		# ind.plot(col='time',vmin=-1,vmax=1)
		
		
		# # Convert the plot to a PNG image in memory
		# buffer = io.BytesIO()
		# plt.savefig(buffer, format='png')
		# buffer.seek(0)
		# base64 = base64.b64encode(buffer.read()).decode('utf-8')
		
		# print(image_base64)
	# print(pl_a)
	print(len(pl_a),len(ml_a))
	
	return jsonify({'pl_a': pl_a,'ml_a':ml_a,'totala':total_area_km2,'message':'null'})
def area( a ,area_per_pixel):
	print(np.sum(a))
	xw=area_per_pixel*np.sum(a)
	print(xw)
	return xw/1000000

def image_to_base64(image):
    plt.figure(figsize=(8,8))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    # print(image_base64)
    return image_base64


		# Display the HTML
		# print(html)
	# return "hello world"
if __name__ == '__main__':
	app.run(debug=True,host='0.0.0.0')