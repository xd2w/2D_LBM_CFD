import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox as msg
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation
import numpy as np
from PIL import Image
import json
import LBM


def donothing():
    print("works")


def error():
    msg.showerror("error", "ERROR")


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("1280x720")
        theme = ttk.Style()
        theme.theme_use("clam")

        self.root.title("2D BLM CFD Simulation")
        self.init_menu()
        self.init_stats()

        self.graph_widget = tk.Frame(self.root, height=500, width=600)
        self.graph_widget.place(x=0, y=0)

        help_text = tk.Label(
            self.graph_widget,
            text="File -> New: open an image to run a simulation\n"
            + "File -> Open: open a calculated result\n"
            + "File -> Save: save the simulated result",
            justify="left",
        )
        help_text.place(x=200, y=150)

        run_frame = tk.Frame(self.root, height=500, width=300)
        run_frame.place(x=710, y=10)

        run_label = tk.Label(
            run_frame, text="Run in realtime without storing the result"
        )
        run_button = tk.Button(run_frame, text="RUN", command=self.run)
        run_label.grid(row=0, column=0, sticky="w")
        run_button.grid(row=0, column=1, sticky="w", padx=20)

        calc_label = tk.Label(
            run_frame, text="Runs the simulation while storing then displays the result"
        )
        calc_button = tk.Button(run_frame, text="CALCULATE", command=self.calculate)
        calc_label.grid(row=1, column=0, sticky="w")
        calc_button.grid(row=1, column=1, sticky="w", padx=20)

        time_duration_label = tk.Label(
            run_frame, text="  - Duration of simulation (number of simulation steps) :"
        )
        self.time_duration = tk.Entry(run_frame, width=4, justify="right")
        time_duration_label.grid(row=2, column=0, sticky="w")
        self.time_duration.grid(row=2, column=1, sticky="w", padx=20)

        self.anim = None

        stop_button = tk.Button(self.root, text="Stop", command=self.pause_animation)
        start_button = tk.Button(self.root, text="Start", command=self.start_animation)
        stop_button.place(x=520, y=500)
        start_button.place(x=520, y=530)

        self.time_scale = tk.Scale(
            self.root, from_=0, to=100, orient="horizontal", length=400, sliderlength=10
        )
        self.time_scale.place(x=600, y=500)

        self.img_path = None
        self.file_path = None

        self.matrix = None

        self.root.mainloop()

    def init_menu(self):
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New", command=self.open_img)
        file_menu.add_command(label="Open", command=self.load_file)
        file_menu.add_command(label="Close", command=donothing)

        file_menu.add_command(label="Export", command=donothing)

        file_menu.add_separator()

        file_menu.add_command(label="Close", command=donothing)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="Undo", command=donothing)

        edit_menu.add_separator()

        edit_menu.add_command(label="Cut", command=donothing)
        edit_menu.add_command(label="Copy", command=donothing)
        edit_menu.add_command(label="Paste", command=donothing)
        edit_menu.add_command(label="Delete", command=donothing)
        edit_menu.add_command(label="Select All", command=donothing)

        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_command(label="Theme", command=donothing)
        view_menu.add_command(label="graph type", command=donothing)

        menu_bar.add_cascade(label="View", menu=view_menu)

        setting_menu = tk.Menu(menu_bar, tearoff=0)
        setting_menu.add_command(label="save format", command=donothing)
        setting_menu.add_command(label="working dir", command=donothing)

        menu_bar.add_cascade(label="Setting", menu=setting_menu)

        self.root.config(menu=menu_bar)

    def init_stats(self):
        stats = tk.Frame(self.root, height=200, width=500)
        stats.place(x=20, y=510)

        self.dimension = tk.StringVar(stats, value="0x0")
        dimension_label = tk.Label(stats, text="Dimension :")
        dimension_value_label = tk.Label(stats, textvariable=self.dimension)
        dimension_label.grid(row=0, column=0, padx=5, pady=5)
        dimension_value_label.grid(row=0, column=1, padx=5, pady=5)

        self.viscosity = tk.StringVar(stats, value=1 / 6.0)
        viscosity_label = tk.Label(stats, text="Viscosity :")
        self.viscosity_entry = tk.Entry(
            stats, textvariable=self.viscosity, width=4, justify="center"
        )
        viscosity_label.grid(row=1, column=0, padx=5, pady=5)
        self.viscosity_entry.grid(row=1, column=1, padx=5, pady=5)

        self.tao = tk.StringVar(stats, value=1)
        tao_label = tk.Label(stats, text="Relaxation time :")
        self.tao_entry = tk.Entry(
            stats, textvariable=self.tao, width=4, justify="center"
        )
        tao_label.grid(row=2, column=0, padx=5, pady=5)
        self.tao_entry.grid(row=2, column=1, padx=5, pady=5)

        self.t_or_v = tk.StringVar(stats, value="tao")

        use_t = tk.Radiobutton(
            stats, text="Use relaxation time", variable=self.t_or_v, value="tao"
        )
        use_v = tk.Radiobutton(
            stats, text="Use viscosity", variable=self.t_or_v, value="viscosity"
        )
        use_t.grid(row=2, column=2, sticky="w", padx=20)
        use_v.grid(row=1, column=2, sticky="w", padx=20)

        self.density = tk.StringVar(stats, value=1)
        density_label = tk.Label(stats, text="Density :")
        self.density_entry = tk.Entry(
            stats, textvariable=self.density, width=4, justify="center"
        )
        density_label.grid(row=3, column=0, padx=5, pady=5)
        self.density_entry.grid(row=3, column=1, padx=5, pady=5)

        apply_button = tk.Button(stats, text="Apply", command=donothing)
        apply_button.grid(row=3, column=3, padx=100)

    def img2shape(self):
        img = Image.open(self.img_path)
        img = np.array(img)

        matrix = np.zeros((len(img[0]), len(img)))

        for y in range(len(img)):
            for x in range(len(img[0])):
                if img[y, x, :2].sum() == 0:
                    matrix[x, y] = True

        self.matrix = matrix

    def open_img(self):
        self.img_path = filedialog.askopenfilename(
            initialdir="~/Pictures",
            title="Select an image",
            filetypes=(("PNG", "*.png*"), ("JPEG", "*.jpg*"), ("all files", "*.*")),
        )

        if self.img_path == "":
            return None

        self.img2shape()
        self.graph(self.matrix)

    def graph(self, values, returnVal=False):
        fig = plt.Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        Q = ax.imshow(np.transpose(values), cmap="Greys")
        graph = FigureCanvasTkAgg(fig, master=self.root)
        graph.draw()

        if returnVal:
            return Q, graph, fig, ax

        self.graph_widget.destroy()
        self.graph_widget = graph.get_tk_widget()
        self.graph_widget.place(x=0, y=0)

    def load_file(self, file_path=None):
        if file_path is None:
            file_path = filedialog.askopenfilename(
                initialdir="~/Documents",
                title="Select the simulation file to load",
                filetypes=(("JSON", "*.json*"), ("all files", "*.*")),
            )

        if file_path == "":
            return None

        with open(file_path, "r") as file:
            data = file.read()

        velocity_dict = json.loads(data)

        meta = velocity_dict[0]
        print(meta)

        self.set_stats(
            tao=meta["tao"],
            viscosity=meta["viscosity"],
            dimension="x".join(list(map(str, meta["dimension"]))),
            rho=meta["density"],
        )
        velocity_dict = velocity_dict[1:]
        self.time_scale.config(to=len(velocity_dict) - 1)
        velocity = np.array(velocity_dict[0]["0"])
        velocity = velocity[:, :, 0]

        fig = plt.Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        q = ax.imshow(np.transpose(velocity))
        graph = FigureCanvasTkAgg(fig, master=self.root)
        graph.draw()

        self.graph_widget.destroy()
        self.graph_widget = graph.get_tk_widget()
        self.graph_widget.place(x=0, y=0)

        self.anim = animation.FuncAnimation(
            fig,
            self.load_file_update,
            frames=len(velocity_dict),
            fargs=(q, velocity_dict),
            interval=50,
            blit=False,
        )
        # gif = animation.PillowWriter(fps=10)
        # self.anim.save("test1.gif", writer=gif)
        tk.mainloop()

    def load_file_update(self, frame, Q, velocity_dict):
        velocity = np.array(velocity_dict[frame][str(frame)])
        Q.set_data(
            np.transpose(
                np.sqrt(
                    np.square(velocity[:, :, 0]) + np.square(velocity[:, :, 1] ** 2)
                )
            )
        )
        self.time_scale.set(frame)
        return Q

    def run(self, calc=False):
        if self.img_path is None:
            msg.showerror("Warning", "there is no image")
            return None

        sim = LBM.Lattice(*self.matrix.shape)
        sim.set_obstacle(self.matrix)

        if self.t_or_v.get() == "tao":
            tao = self.tao_entry.get()
            sim.set_tao(float(tao))
        elif self.t_or_v.get() == "viscosity":
            visc = self.viscosity_entry.get()
            sim.set_viscosity(float(visc))
        else:
            error()

        if calc is True:
            sim.save_to()
            duration = int(self.time_duration.get())
            sim.simulate(duration)
            self.load_file("temp.json")

            return None

        fig = plt.Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        heatmap = ax.imshow(np.transpose(self.matrix * 0.5))
        graph = FigureCanvasTkAgg(fig, master=self.root)
        graph.draw()

        self.graph_widget.destroy()
        self.graph_widget = graph.get_tk_widget()
        self.graph_widget.place(x=0, y=0)

        self.anim = animation.FuncAnimation(
            fig, self.update_run, fargs=(heatmap, sim), interval=50, blit=False
        )

        tk.mainloop()

    def update_run(self, num, q, sim):
        sim.simulate(1)
        q.set_data(
            np.transpose(
                np.sqrt(
                    sim.bulk_velocity[:, :, 0] ** 2 + sim.bulk_velocity[:, :, 1] ** 2
                )
            )
        )
        self.time_scale.set(num)
        return q

    def calculate(self):
        self.run(calc=True)

    def pause_animation(self):
        if self.anim is not None:
            self.anim.event_source.stop()

    def start_animation(self):
        if self.anim is not None:
            self.anim.event_source.start()

    def set_stats(self, tao=None, viscosity=None, dimension=None, rho=None):
        if tao is not None:
            self.tao.set(tao)
        if viscosity is not None:
            self.viscosity.set(viscosity)
        if dimension is not None:
            self.dimension.set(dimension)
        if rho is not None:
            self.density.set(rho)


if __name__ == "__main__":
    a = GUI()
