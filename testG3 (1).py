import numpy as np
import pandas as pd
from customtkinter import *

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Toplevel
from tkinter import messagebox
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from mpl_toolkits import mplot3d




# Charger le dataset d'entraînement iris depuis un fichier CSV
iris = pd.read_csv("https://gist.github.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")

iris_setosa = iris[iris['variety'] == 'Setosa']
iris_versicolor = iris[iris['variety'] == 'Versicolor']
iris_virginica = iris[iris['variety'] == 'Virginica']

# Features : variables explicatives
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

# Créer une instance de modèle SVM
model = None

def classify_iris(sepal_length, sepal_width, petal_length, petal_width):
    global model

    # Récupérer les valeurs saisies
    selected_kernel = kernel_var.get()
    selected_C = float(C_entry.get())
    selected_gamma = float(gamma_entry.get())

    # Créer le modèle SVM en fonction des choix de l'utilisateur
    model = svm.SVC(C=selected_C, kernel=selected_kernel, gamma=selected_gamma)

    # Entraîner le modèle SVM sur toutes les données
    model.fit(X, y)

    # Créer une liste des valeurs saisies pour l'iris à classer
    iris_to_classify = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Classification de l'iris saisi
    prediction = model.predict(iris_to_classify)
    
    # Calculer l'évaluation du modèle
    y_pred = model.predict(X)
    evaluation = accuracy_score(y, y_pred)

    # Calculer le rapport de classification
    classification_rep = classification_report(y, y_pred)

    # Créer une nouvelle fenêtre pour afficher le résultat
    
    result_window = CTkToplevel(root)
    result_window.geometry("500x500")
    result_window.title("classification result")
    result_window.resizable(False, False)
    # Créer un label pour afficher le résultat
    result_label = CTkLabel(master=result_window,font=("Arial", 16), text_color="#FFCC70" ,text=f"classification of iris entered is : {prediction}")
    result_label.place(x=100, y=50)
    
    #Évaluation du modèle
    evaluation_label = CTkLabel(master=result_window,font=("Arial", 16), text_color="#FFCC70", text=f"Model Evaluation  : {evaluation}")
    evaluation_label.place(x=150, y=150)
    
    #Rapport de classification
    classification_label = CTkLabel(master=result_window,font=("Arial", 15), text_color="#FFCC70", text=f"Classification rapport :\n{classification_rep}")
    classification_label.place(x=100, y=250)
    
    
# Interface graphique

root = CTk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()

x = (width - width) // 2
Y = (height - height) // 2
root.geometry(f"{width}x{height}+{x}+{Y}")
#root.config(background="pink")

root.geometry("1150x600")
set_appearance_mode("dark") 
root.title("Classification of an iris with SVM algorithm")

frame = CTkFrame(master=root ,width=590, height=500,border_color="#FFCC70", border_width=2, corner_radius=20, fg_color="transparent")
frame.place(x=530, y=50)
labsvm = CTkLabel(master=root, text="SVM Algorithm", font=("Arial", 25), text_color="#FFCC70")
labsvm.place(x=150, y=30)

# Création d'une figure matplotlib
fig = Figure(figsize=(7, 6), dpi=100)
ax = fig.add_subplot(111)
ax.scatter(iris_setosa['sepal.length'], iris_setosa['sepal.width'], color='red', label='Setosa')
ax.scatter(iris_versicolor['sepal.length'], iris_versicolor['sepal.width'], color='blue', label='Versicolor')
ax.scatter(iris_virginica['sepal.length'], iris_virginica['sepal.width'], color='green', label='Virginica')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_title('Iris Sepal Length vs Sepal Width')
ax.legend()

# Création d'un widget Canvas pour afficher la figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().place(x=10, y=150)

# Liste des noyaux disponibles
kernels = ['linear', 'rbf', 'poly']

# Choix du noyau SVM
label_kernel = CTkLabel(master=root, text="Kernel :")
label_kernel.place(x=740, y=400)

kernel_var = tk.StringVar(value="rbf")
kernel_combobox = CTkComboBox(master=root,width=100, values=kernels, variable=kernel_var)
kernel_combobox.place(x=790, y=404)

# Valeur de C
label_C = CTkLabel(master=root, text="value of C :")
label_C.place(x=570, y=400)
C_entry = CTkEntry(master=root,width=70, height=6 )
C_entry.place(x=650, y=404)

# Valeur de gamma
label_gamma = CTkLabel(master=root, text="gamma value :")
label_gamma.place(x=920, y=400)
gamma_entry = CTkEntry(master=root,width=70, height=6)
gamma_entry.place(x=1010, y=404)

# Saisie des valeurs de l'iris à classer
label_values = CTkLabel(master=root, text="Entry the Iris values to classify :",font=("Arial", 30), text_color="#FFCC70")
label_values.place(x=590, y=90)

sepal_length_label = CTkLabel(master=root, text='Sepal Length')
sepal_length_label.place(x=700, y=165)

sepal_width_label = CTkLabel(master=root, text='Sepal Width')
sepal_width_label.place(x=700, y=225)

petal_length_label = CTkLabel(master=root, text='Petal Length')
petal_length_label.place(x=700, y=285)

petal_width_label = CTkLabel(master=root, text='Petal Width')
petal_width_label.place(x=700, y=345)

sepal_length_entry = CTkEntry(master=root)
sepal_length_entry.place(x=800, y=165)

sepal_width_entry = CTkEntry(master=root)
sepal_width_entry.place(x=800, y=225)

petal_length_entry = CTkEntry(master=root)
petal_length_entry.place(x=800, y=285)

petal_width_entry = CTkEntry(master=root)
petal_width_entry.place(x=800, y=345)

# Bouton pour classifier l'iris saisi
classify_button = CTkButton(master=frame, text="Classify iris", corner_radius=32, fg_color="transparent", hover_color="#4158D0", border_color="#FFCC70", border_width=2,width=300,height=50, command=lambda: classify_iris(
    float(sepal_length_entry.get()), float(sepal_width_entry.get()),
    float(petal_length_entry.get()), float(petal_width_entry.get())
))
classify_button.place(relx=0.5, rely=0.9, anchor="center")


root.mainloop()