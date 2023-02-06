import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, AnyStr
from typing_extensions import Self
import sympy as sy
import sympy.vector as sv
from dataclasses import dataclass
from scipy import signal

t = np.linspace(-np.pi, np.pi, 100)

amplitudes = [1, 0.1, 0.2, 0.3, 0.5, 0.8, 1.3, 2.1, 3.4]
styles = ['solid', 'solid', 'dashed', 'dotted', 'solid', 'dashed', 'dotted', 'solid', 'dashed']

@dataclass
class Sym2D:
    _0: 'symbolic'
    _1: 'symbolic'

    def __str__(self):
        return f'[{self._0}, {self._1}]'

    def __getitem__(self, key):
        match key:
            case 0: return self._0
            case 1: return self._1
            case _: raise IndexError
    
    def div(self, value):
        if type(value) == Sym2D:
            raise Exception('unimplemented')
        return Sym2D(self._0 / value, self._1 / value)
    
    def diff(self, var):

        # gradient case
        if type(var) == Sym2D:
            raise Exception('unimplemented')
        
        # partial derivative case
        return Sym2D(sy.diff(self._0, var), sy.diff(self._1, var))

    def norm_gradient(self, s: Self, x):
        """norm of gradient with respect to s vector, with respect to x"""
        
        return self.diff(x).norm() / s.diff(x).norm()

    def norm(self):
        return sy.sqrt(self._0**2 + self._1**2)

    def normalize(self):
        return self.div(self.norm())
    
    def simplify(self):
        return Sym2D(sy.simplify(self._0), sy.simplify(self._1))

def curvature1D(x, f):
    return sy.simplify(sy.sqrt(sy.Derivative(f, (x, 2))**2/(sy.Derivative(f, x)**2 + 1)**2)/sy.sqrt(sy.Derivative(f, x)**2 + 1))

def analyze_sym(fs) -> None:
    x = sy.symbols('x')

    # find curvature of fs
    # r = Sym2D(x, fs)
    # rd = r.diff(x)
    # T = rd.div(rd.norm()).normalize().simplify()
    # fcs = sy.simplify(T.norm_gradient(r, x))
    
    fcs = curvature1D(x, fs)
    fds = sy.diff(fs, x)
    fdds = sy.diff(fs, x, 2)

    f = sy.lambdify(x, fs, "numpy")
    fc = sy.lambdify(x, fcs, "numpy")
    fd = sy.lambdify(x, fds, "numpy")
    fdd = sy.lambdify(x, fdds, "numpy")

    ft = f(t)
    fct = fc(t)
    fdt = fd(t)
    fddt = fdd(t)

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()
    fig.set_size_inches(12,9)
    axs[0,0].set_ylim(-1.1,1.1)
    axs[0,0].axhline(y = 0, color='black')
    axs[0,0].axvline(x = 0, color='black')
    axs[0,0].fill_between(t, -2, 2, where= ft < -0.975, color='red', alpha=0.2)
    axs[0,0].fill_between(t, -2, 2, where= ft > 0.975, color='red', alpha=0.2)
    axs[0,0].imshow([fct], aspect='auto', extent=[-np.pi,np.pi,-1.1,1.1], cmap='Greys', alpha=0.3)
    axs[0,0].grid()
    axs[0,0].plot(t, fdt, color='black', linewidth=0.7)
    axs[0,0].plot(t, fddt, color='black', linewidth=0.7)
    axs[0,0].plot(t, ft, color='red', linewidth=2)
    
    axs[0,1].axhline(y = 0, color='black')
    axs[0,1].axvline(x = 0, color='black')
    axs[0,1].grid()
    sine = np.sin(np.linspace(0, 4 * np.pi, 128))
    axs[1,1].set_xscale('log')
    for i, a in enumerate(amplitudes):
        axs[0,1].plot(
            t, f(np.sin(t) * a), 
            linewidth= 2 if i == 0 else 0.7,
            color= 'red' if i == 0 else 'black',
            linestyle= styles[i])
        spectrum = np.fft.rfft(f(a * sine))
        spectrum -= np.min(spectrum)
        axs[1,1].plot(
            spectrum,
            linewidth= 2 if i == 0 else 0.7,
            color= 'red' if i == 0 else 'black',
            linestyle= styles[i])
    
    ncycles = 64
    samples_per_cycle = 64
    t_a = np.linspace(0,3, ncycles * samples_per_cycle)
    t_phi = np.linspace(0, ncycles * 2 * np.pi, ncycles * samples_per_cycle)
    chirp = f(t_a * np.sin(t_phi))
    fft = signal.stft(chirp)
    axs[1,0].specgram(chirp)
    
    
    
    



    


def _test():
    #analyze_f(lambda x: np.clip(x, -1, 1))
    x = sy.symbols('x')
    analyze_sym(sy.sin(x))
    plt.show()

if __name__ == "__main__":
    _test()