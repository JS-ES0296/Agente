import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

class AgenteReactivo:
    def __init__(self):
        self.archivo = None
        

    def cargar_archivo(self, archivo):
        self.archivo = archivo

    def procesar_archivo(self):
        # Cargar los datos del archivo CSV
        datos = pd.read_csv(self.archivo)
        dt = datos[datos["U1[V]"]< 100].index
        dtf = datos.drop(dt)
        dc = dtf[dtf["I1[A]"]==0].index
        data = dtf.drop(dc)
        # Separar los datos en variables predictoras y etiquetas
        x = data.iloc[:,2:3].values  
        y = data.iloc[:, 3].values   

        # Dividir los datos en conjuntos de entrenamiento y prueba
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=19)
        x_train
        x_test
        # Crear el clasificador de Random Forest
        from sklearn.ensemble import RandomForestRegressor 
        regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
        regressor.fit(x_train, y_train) 

        # Visualizar los resultados
        X_grid = np.arange(min(x_train), max(x_train),.001)  
        X_grid = X_grid.reshape((len(X_grid), 1)) 
        fig, ax = plt.subplots()
        ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed') 
        ax.scatter(x_train, y_train, color = 'purple', label = 'Corriente')   
        ax.plot(X_grid, regressor.predict(X_grid),color = 'orange', linestyle = 'dashed', label = 'Voltaje')  
        ax.legend(loc = 'upper right')
        # Predecir las etiquetas de los datos de prueba
        y_pred = regressor.predict(x_test)

        # Evaluar la precisión del modelo
        precision = regressor.score(x_test, y_test)
        ax.text(0.5, 0.5, f'Precisión del modelo: {precision:.5f}', transform=ax.transAxes, verticalalignment='bottom')
        plt.title('Voltaje vs Corriente') 
        plt.xlabel('U1[V]') 
        plt.ylabel('I1[A]') 
        plt.show()


        # print('Precisión del modelo:', precision)


class Interfaz(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.crear_widgets()
    def crear_widgets(self):
        self.cargar_archivo_button = tk.Button(self, text="Cargar archivo", command=self.cargar_archivo)
        self.cargar_archivo_button.pack(side="top")

        self.procesar_archivo_button = tk.Button(self, text="Procesar archivo", command=self.procesar_archivo)
        self.procesar_archivo_button.pack(side="top")

        self.salir = tk.Button(self, text="Salir", fg="red", command=self.master.destroy)
        self.salir.pack(side="bottom")

    def cargar_archivo(self):
        archivo = filedialog.askopenfilename()
        self.agente_reactivo = AgenteReactivo()
        self.agente_reactivo.cargar_archivo(archivo)

        
    def cerrar_ventana_progreso(self):
        self.ventana_progreso.destroy()

    def iniciar(self):
        self.mainloop()


    def procesar_archivo(self):
         # Crear la ventana de progreso
        self.ventana_progreso = tk.Toplevel(self)
        self.ventana_progreso.title("Procesando archivo")
        self.ventana_progreso.geometry("300x100")
        self.label_progreso = tk.Label(self.ventana_progreso, text="Procesando...")
        self.label_progreso.pack(pady=10)
        self.progressbar = ttk.Progressbar(self.ventana_progreso, orient="horizontal", length=200, mode="indeterminate")
        self.progressbar.pack(pady=10)
        self.progressbar.start()

        # Programar el cierre automático de la ventana de progreso después de 5 segundos
        self.after(1000, self.cerrar_ventana_progreso)

        self.agente_reactivo.procesar_archivo()


root = tk.Tk()
app = Interfaz(master=root)
app.mainloop()
