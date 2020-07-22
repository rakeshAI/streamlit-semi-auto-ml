# Core pkgs
import streamlit as st

#EDA pkgs 
import pandas as pd 
import numpy as np 
import seaborn as sns 
#data viz pkgs
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib 
matplotlib.use('Agg')

#ML pkgs
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def main():
	"""Semi Auto ML App with streamlit """
	image = Image.open("semiautoml.jpg")
	st.image(image, use_column_width=True)


	#st.title("Semi Auto ML App")

	activities = ["EDA", "Plot", "Model Building", "About"]

	choice = st.sidebar.selectbox("Select Activity", activities)

	if choice == 'EDA':
		st.title("Exploratory Data Analysis")

		data = st.file_uploader("Upload Dataset", type=["csv","txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox("Show shape"):
				st.write(df.shape)

			if st.checkbox("Show Coulmns"):
				all_columns = df.columns.to_list()
				st.write(all_columns)

			if st.checkbox("Select Coulmns To Show"):
				all_columns = df.columns.to_list()
				selected_columns = st.multiselect("Select Columns", all_columns)
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Show Summary"):
				st.write(df.describe())

			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts())




	elif choice == 'Plot':
		st.title("Data Visualization")
		data = st.file_uploader("Upload Dataset", type=["csv","txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox("Correlation with Seaborn"):
				st.write(sns.heatmap(df.corr(),annot=True))
				st.pyplot()

			if st.checkbox("Pie Chart"):
				all_columns = df.columns.to_list()
				columns_to_plot = st.selectbox("Select 1 Column", all_columns)
				pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pie_plot)
				st.pyplot()

			if st.checkbox("Select Chart"):
				all_columns_names = df.columns.tolist()
				type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
				selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

				if st.button("Generate Plot"):
					st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

					if type_of_plot == 'area':
						cust_data = df[selected_columns_names]
						st.area_chart(cust_data)

					elif type_of_plot == 'bar':
						cust_data = df[selected_columns_names]
						st.bar_chart(cust_data)

					elif type_of_plot == 'line':
						cust_data = df[selected_columns_names]
						st.line_chart(cust_data)

					elif type_of_plot:
						cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
						st.write(cust_plot)
						st.pyplot()

	elif choice == 'Model Building':
		st.title("Building ML Model")
		data = st.file_uploader("Upload Dataset", type=["csv","txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			#Model Building
			X = df.iloc[:,0:-1]
			Y = df.iloc[:,-1]
			seed = 0

			# Model
			models = []
			models.append(("LR",LogisticRegression()))
			models.append(("LDA",LinearDiscriminantAnalysis()))
			models.append(("KNN",KNeighborsClassifier()))
			models.append(("CART", DecisionTreeClassifier()))
			models.append(("NB", GaussianNB()))
			models.append(("SVM", SVC()))

			#List
			model_names = []
			model_mean = []
			model_std = []
			all_models = []
			scoring = 'accuracy'

			for name,model in models:
				kfold = model_selection.KFold(n_splits=10,random_state=seed)
				cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
				model_names.append(name)
				model_mean.append(cv_results.mean())
				model_std.append(cv_results.std())

				accuracy_result = {"model_name":name, "model_accuracy":cv_results.mean(), "standard_deviation":cv_results.std()}
				all_models.append(accuracy_result)

			if st.checkbox("Metrics as Table"):
				st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std), columns=["Model Name", "Model Accuracy", "Standard Deviation"]))

			if st.checkbox("Metrics as JSON"):
				st.json(all_models)


	elif choice == 'About':
		st.title("About")
		st.text("Using streamlit and ML packages created a simple semi auto machine learning web app")
		st.text("Author:Jammula Rakesh Kumar")

if __name__ == '__main__':
	main()

