import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CartPoleVisualizer:
    def __init__(self, history, config):
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.grid()
        
        # Create cart and pole objects
        self.cart, = self.ax.plot([], [], 'o-', lw=2)
        self.pole, = self.ax.plot([], [], '-', lw=2)

        self.history = history
        self.config = config

        self.out_dir = "./out/"  # Path to save output gif and plot

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
    
    # Initialize animation
    def init(self):
        self.cart.set_data([], [])
        self.pole.set_data([], [])
        return self.cart, self.pole
    
    # Update animation for each frame
    def update(self, frame):
        x     = self.history[frame, 0]
        theta = self.history[frame, 2]
        
        self.cart.set_data([x - 0.5, x + 0.5], [0, 0])
        self.pole.set_data([x, x + self.config.l * np.sin(theta)], [0, -self.config.l * np.cos(theta)])
        return self.cart, self.pole

    def gen_animation(self):
        # Create animation
        ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.history), interval=self.config.dt*1000)

        gif_writer = animation.ImageMagickWriter(fps=20)
        # Uncomment the line below to save the animation
        ani.save(self.out_dir+'cart_pole_animation.gif',writer='pillow')
        
        plt.show()

    def plot(self, states, actions):
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        axs[0].plot(actions, 'r')
        axs[1].plot(states[:,2]*180/np.pi, 'b')
        plt.savefig(self.out_dir+"cart_pole_actions.png")
        plt.close()
