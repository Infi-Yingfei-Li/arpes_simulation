import os
import h5py
import numpy as np
import scipy
from scipy import integrate
import scipy.fftpack as ft
import copy
import matplotlib.pyplot as plt
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing

#import sympy
#import sympy.physics.wigner

class radial_schrodinger_wave:
    '''
    calculate the radial part of the wave function under Hartree-Fock central potential
    '''
    def __init__(self, config):
        '''
        Initialize the class.
        params:
            config: dict, class configuration, with following keys:
                - config["Z"]: int, the nuclear charge of the atom
                - config["l"]: int, the angular momentum of the atom
        '''
        self.Z = config['Z']
        self.l = config['l']
        HF_potential = h5py.File('HF_potential.hdf5','r')
        self.U_data = np.array([list(HF_potential[str(self.Z)]['x'][:]), list(HF_potential[str(self.Z)]['U'][:])]).T
        self.U_data[:,0] *=  0.88534138*np.power(self.Z, -1/3)
        HF_potential.close()
    
    def P_func_and_delta(self, E):
        '''
        Params: 
            E: kinetic energy, in the units of Rydberg
        return:
            if E>0:
                self.P: the radial wave function of bound state
            else:
                self.P, self.delta_l: the radial wave function and the phase shift of scattering state
        '''
        if E<0:
            r_start = self.U_data[0,0]; r_end = 100*self.U_data[-1,0]; r_delta = 0.001
            r_mesh = np.arange(r_start, r_end, r_delta)[::-1]
            sol = [[1,0]]
            
            for r in r_mesh:
                P, dPdr = sol[-1][0], sol[-1][1]
                P -= r_delta*dPdr
                dPdr -= r_delta*(-(self._V(r)+E-self.l*(self.l+1)/np.power(r,2))*P)                
                sol.append([P, dPdr])
                sol = self._semi_normalization(sol)
            
            P = np.array(sol)[1:len(sol), 0]
            
            def integrand(x):
                return np.power(np.interp(x, r_mesh[::-1], P[::-1]),2)
            
            self.r_mesh = r_mesh[::-1]
            self.P = -(P/np.sqrt(integrate.quad(integrand, r_start, r_end, limit=10000)[0]))[::-1]            
            return self.P
        
        if E>0:
            r_start = self.U_data[0,0]; r_end = 100*self.U_data[-1,0]; r_delta = 0.001
            r_mesh = np.arange(r_start, r_end, r_delta)
            
            if self.l == 0:
                sol = [[r_start, 1]]
            if self.l == 1:
                sol = [[np.power(r_start, 2), 2*r_start]]
            if self.l == 2:
                sol = [[np.power(r_start, 3), 3*np.power(r_start, 2)]]
            if self.l == 3:
                sol = [[np.power(r_start, 4), 4*np.power(r_start, 3)]]
            
            for r in r_mesh:
                P, dPdr = sol[-1][0], sol[-1][1]
                P += r_delta*dPdr
                dPdr += r_delta*(-(self._V(r)+E-self.l*(self.l+1)/np.power(r,2))*P)
                sol.append([P, dPdr])
                sol = self._semi_normalization(sol)
            
            P = np.array(sol)[0:(len(sol)-1), 0]
            norm_const = np.power(np.pi, -1/2)*np.power(E, -1/4)/np.max(np.abs(P[round(0.9*len(r_mesh)):len(r_mesh)]))
            
            self.r_mesh = r_mesh; self.P = norm_const*P
            self.delta_l = self._delta(r_mesh[round(0.9*len(r_mesh)):len(r_mesh)], P[round(0.9*len(r_mesh)):len(r_mesh)], E)
                        
            return self.P, self.delta_l
    
    def _V(self, r):
        '''
        params:
            r: float, radias
        return:
            Hartree-Fock central potential at r        
        '''
        return (2*self.Z/r)*np.interp(r, self.U_data[:,0], self.U_data[:,1], left=self.U_data[0,1], right=self.U_data[-1,1])
    
    def _x2r(self, x):
        mu = 0.88534138*np.power(self.Z, -1/3)
        return mu*x
    
    def _delta(self, x, y, E):
        '''
        Calculate the phase shift of scatter state at infinity by least-square fit.        
        '''
        
        y_normalized = y/np.max(np.abs(y))
        delta_list = np.arange(0, 2*np.pi, 0.1)
        RSS = []
        fit_value_hist = []
        for delta in delta_list:
            fit_value = [np.sin(np.sqrt(E)*x_j - self.l*np.pi/2 - np.power(E, -1/2)*np.log(2*np.sqrt(E)*x_j)+delta) for x_j in x]
            RSS.append(np.mean([np.power(y_normalized[j]-fit_value[j], 2) for j in range(len(x))]))
            fit_value_hist.append(fit_value)
            
        delta_list_fine = np.arange(delta_list[np.argmin(RSS)]-0.1*np.pi, delta_list[np.argmin(RSS)]+0.1*np.pi, 0.01)
        RSS_fine = []
        fit_value_hist_fine = []
        for delta in delta_list_fine:
            fit_value = [np.sin(np.sqrt(E)*x_j - self.l*np.pi/2 - np.power(E, -1/2)*np.log(2*np.sqrt(E)*x_j)+delta) for x_j in x]
            RSS_fine.append(np.mean([np.power(y_normalized[j]-fit_value[j], 2) for j in range(len(x))]))
            fit_value_hist_fine.append(fit_value)
        
        return delta_list_fine[np.argmin(RSS_fine)]
    
    def _semi_normalization(self, ar):
        '''
        Normalize wave function to avoid overfloating error during integration.
        '''
        maximum = np.max(np.abs(ar[-1]))
        if maximum > 1e3:
            return (np.array(ar)/maximum).tolist()
        else:
            return ar


config = {'Z': 33,
         'orbital': '4pz'}

class matrix_element_scenario_1:
    '''
    Calculate atomic photoemission matrix element under dipole approximation with
    initial state: bound state under Hartree-Fock central potential
    final state: scattering state under Hartree-Fock central potential
    '''
    def __init__(self, config):
        '''
        Initialize class
        params:
            config: dict, class configuration, with following keys:
                - config["Z"]: int, the nuclear charge of the atom
                - config["orbital"]: str, the conventional name of orbital, eg. "1s", "2px", "2py", "2pz",\
                                    "3dxy", "3dxz", "3dyz", "3dz2", "3dx2-y2", etc.
        '''
        self.config = config
        self.Z = config['Z']
        self.orbital = config['orbital']
        
        HF_potential = h5py.File('HF_potential.hdf5','r')
        self.U_data = np.array([list(HF_potential[str(self.Z)]['x'][:]), list(HF_potential[str(self.Z)]['U'][:])]).T
        self.U_data[:,0] *=  0.88534138*np.power(self.Z, -1/3)
        try:
            orbital_idx = [j for j in range(len(HF_potential[str(self.Z)]['orbital'][:])) if HF_potential[str(self.Z)]['orbital'][j].decode('utf-8') == self.orbital[0:2]][0]
        except:
            orbital_idx = [j for j in range(len(HF_potential[str(self.Z)]['orbital'][:])) if HF_potential[str(self.Z)]['orbital'][j] == self.orbital[0:2]][0]
        self.E_nl = HF_potential[str(self.Z)]['energy'][orbital_idx]
        HF_potential.close()
        
        if self.orbital[1] == 's':
            self.l = 0
        if self.orbital[1] == 'p':
            self.l = 1
        if self.orbital[1] == 'd':
            self.l = 2
        
        self.r_mesh_start = self.U_data[0,0]
        self.r_mesh_end = 100*self.U_data[-1,0]
        self.r_mesh_delta = 0.001
        self.prefactor = 4*np.pi*np.power(0.52918,2)/(3*13704)
        
    def matrix_element_core(self, hn, theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2, k_x=0.01, k_y=0.01):
        '''
        Calculate matrix element at given photon energy and in-plane momentum.
        params:
            hn: float, photon energy, in the unit of eV
            theta_epsilon, phi_epsilon: float, in the unit of rad. \
                                        The direction of electric vector of incident light.
                                        At Stanford Syncrotron Radiation Lightsource Beamline 5-2,
                                        LH (p) polarization: (theta_epsilon, phi_epsilon)=2*np.pi/9, np.pi/2
                                        LV (s) polarization: (theta_epsilon, phi_epsilon)=np.pi/2, 0
            k_x, k_y: float, the in-plane momentum, in the units of Angstrom. The corresponding photoemission angle are calculated correspondingly.
            
        return:
            (cross_section, R_0, R_2, delta_0, delta_2)
            where:
                cross_section: the matrix element 
                R_0, R_2: the radial integral of the two l'=l\pm 1 channels
                delta_0, delta_2: the phase shift of final state of the two l'=l\pm 1 channels
        '''
        print('--- calculating matrix element at hn = ', hn, ' eV, k_x=', k_x, ', k_y=', k_y, ' ---')
        theta_k, phi_k = self._momentum2angle(hn, k_x, k_y)
        epsilon_x = np.sin(theta_epsilon)*np.cos(phi_epsilon)
        epsilon_y = np.sin(theta_epsilon)*np.sin(phi_epsilon)
        epsilon_z = np.cos(theta_epsilon)
        
        if self.l == 0:
            R_1, delta_1 = self._radial_int_and_delta(hn, self.l, self.l+1)
            k = 0.512*np.sqrt(hn + self.E_nl*13.605693)
            k_z = np.sqrt(k**2-k_x**2-k_y**2)
            return (self.prefactor*hn*4*np.pi*np.power(R_1, 2)*np.power(epsilon_x*k_x+epsilon_y*k_y+epsilon_z*k_z, 2), R_1, delta_1)
        
        if self.l == 1:
            R_0, delta_0 = self._radial_int_and_delta(hn, self.l, self.l-1)
            R_2, delta_2 = self._radial_int_and_delta(hn, self.l, self.l+1)
            
            if self.orbital[1:] == 'px':
                X_0 = 2*np.sqrt(2*np.pi)*R_0*np.sqrt(1/3)*epsilon_x
                X_2 = 2*np.sqrt(np.pi)*R_2*(-np.sqrt(3/4)*np.power(np.sin(theta_k), 2)*(epsilon_x*np.cos(2*phi_k)\
                      +epsilon_y*np.sin(2*phi_k))+epsilon_x*np.sqrt(1/12)*(3*np.power(np.cos(theta_k),2)-1)\
                      -epsilon_z*np.sqrt(3)*np.sin(theta_k)*np.cos(theta_k)*np.cos(phi_k))
            
            if self.orbital[1:] == 'py':
                X_0 = 2*np.sqrt(np.pi)*R_0*np.sqrt(1/3)*epsilon_y
                X_2 = 2*np.sqrt(np.pi)*R_2*(np.sqrt(3/4)*np.power(np.sin(theta_k),2)*(-epsilon_x*np.sin(2*phi_k)\
                        +epsilon_y*np.cos(2*phi_k))+epsilon_y*np.sqrt(1/12)*(3*np.power(np.cos(theta_k),2)-1)\
                        -epsilon_z*np.sqrt(3)*np.sin(theta_k)*np.cos(theta_k)*np.sin(phi_k))
                
            if self.orbital[1:] == 'pz':
                X_0 = 2*np.sqrt(2*np.pi)*R_0*np.sqrt(1/6)*epsilon_z
                X_2 = 2*np.sqrt(2*np.pi)*R_2*(np.sqrt(3/2)*np.sin(theta_k)*np.cos(theta_k)\
                        *(-epsilon_x*np.cos(phi_k)-epsilon_y*np.sin(phi_k))\
                        - epsilon_z*np.sqrt(1/6)*(3*np.power(np.cos(theta_k),2)-1))
            
            return (self.prefactor*hn*(np.power(X_0, 2) + np.power(X_2, 2) + 2*X_0*X_2*np.cos(delta_2-delta_0)), R_0, R_2,\
                    delta_0, delta_2)
        
        if self.l == 2:
            R_1, delta_1 = self._radial_int_and_delta(hn, self.l, self.l-1)
            R_3, delta_3 = self._radial_int_and_delta(hn, self.l, self.l+1)
            
            if self.orbital[1:] == 'dxy':
                X_1 = 2*np.sqrt(np.pi)*R_1*np.sqrt(3/5)*np.sin(theta_k)*(epsilon_x*np.sin(phi_k)+epsilon_y*np.cos(phi_k))
                X_3 = -2*np.sqrt(np.pi)*R_3*(-np.sqrt(3/80)*np.sin(phi_k)*(5*np.power(np.cos(theta_k),2)-1)*(epsilon_x*np.sin(phi_k)\
                        + epsilon_y*np.cos(phi_k))+np.sqrt(15/16)*np.power(np.sin(theta_k),3)*(epsilon_x*np.sin(3*phi_k)-epsilon_y*np.cos(3*phi_k))\
                        + epsilon_z*np.sqrt(15/4)*np.power(np.sin(theta_k),2)*np.cos(theta_k)*np.sin(2*phi_k))
                
            if self.orbital[1:] == 'dxz':
                X_1 = 2*np.sqrt(np.pi)*R_1*(np.sqrt(3/5)*(epsilon_x*np.cos(theta_k)+epsilon_z*np.sin(theta_k)*np.cos(phi_k)))
                X_3 = -2*np.sqrt(np.pi)*R_3*(np.sqrt(15/4)*np.power(np.sin(theta_k),2)*np.cos(theta_k)*(epsilon_x*np.cos(2*phi_k)\
                    + epsilon_y*np.sin(2*phi_k))-epsilon_x*np.sqrt(3/20)*(5*np.power(np.cos(theta_k),3)-3*np.cos(theta_k))\
                    + epsilon_z*np.sqrt(3/5)*np.sin(theta_k)*(5*np.power(np.cos(theta_k),2)-1)*np.cos(phi_k))
                
            if self.orbital[1:] == 'dyz':
                X_1 = 2*np.sqrt(np.pi)*R_1*(np.sqrt(3/5)*(epsilon_y*np.cos(theta_k)+epsilon_z*np.sin(theta_k)*np.sin(phi_k)))
                X_3 = -2*np.sqrt(np.pi)*R_3*(np.sqrt(15/4)*np.power(np.sin(theta_k),2)*np.cos(theta_k)*(epsilon_x*np.sin(2*phi_k)\
                    - epsilon_y*np.cos(2*phi_k))-epsilon_y*np.sqrt(3/20)*(5*np.power(np.cos(theta_k),3)-3*np.cos(theta_k))\
                    + epsilon_z*np.sqrt(3/5)*np.sin(theta_k)*(5*np.power(np.cos(theta_k),2)-1)*np.sin(phi_k))
                
            if self.orbital[1:] == 'dx2-y2':
                X_1 = -2*np.sqrt(np.pi)*R_1*np.sqrt(3/5)*np.sin(theta_k)*(-epsilon_x*np.cos(phi_k)+epsilon_y*np.sin(phi_k))
                X_3 = -2*np.sqrt(np.pi)*R_3*(np.sqrt(3/80)*np.sin(theta_k)*(5*np.power(np.cos(theta_k),2)-1)*(-epsilon_x*np.cos(phi_k)+epsilon_y*np.sin(phi_k))\
                                            +np.sqrt(15/16)*np.power(np.sin(theta_k),3)*(epsilon_x*np.cos(3*phi_k)+epsilon_y*np.sin(3*phi_k))\
                                            +epsilon_z*np.sqrt(15/4)*np.power(np.sin(theta_k),2)*np.cos(theta_k)*np.cos(2*phi_k))
                
            if self.orbital[1:] == 'dz2':
                X_1 = -2*np.sqrt(2*np.pi)*R_1*(np.sqrt(1/10)*np.sin(theta_k)*(epsilon_x*np.cos(phi_k)+epsilon_y*np.sin(phi_k))\
                                               -epsilon_z*np.sqrt(2/5)*np.cos(theta_k))
                X_3 = -2*np.sqrt(2*np.pi)*R_3*(np.sqrt(9/40)*np.sin(theta_k)*(5*np.power(np.cos(theta_k),2)-1)*(epsilon_x*np.cos(phi_k)+epsilon_y*np.sin(phi_k))\
                                              +epsilon_z*np.sqrt(9/40)*(5*np.power(np.cos(theta_k),3)-3*np.cos(theta_k)))
                
            return (self.prefactor*hn*(np.power(X_1, 2) + np.power(X_3, 2) + 2*X_1*X_3*np.cos(delta_3-delta_1)), R_1, R_3,\
                    delta_1, delta_3)
    
    def matrix_element_hn_dependence(self, hn_range, \
                                     theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2, k_x=0.01, k_y=0.01,\
                                    save_file = 'matrix_element.csv'):
        '''
        Calculate photon-energy dependence of cross-section
        params:
            hn_range: list, list of photon energy to calculate
            theta_epsilon, phi_epsilon, k_x, k_y: same as the definition in function self.matrix_element_core
            save_file: the saved file name
            
        return:
            (hn_range, matrix_element_list, R_0_list, R_2_list, delta_0_list, delta_2_list)
            hn_range: a list of photon energy to calculate
            matrix_element_list: the corresponding list of cross-section
            R_0_list, R_2_list, delta_0_list, delta_2_list: the corresponding list of R_0, R_2, delta_0, delta_2, same as the definition in function matrix_element_core
        '''
        result_list = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.matrix_element_core)(hn, theta_epsilon, phi_epsilon, k_x, k_y) for hn in hn_range)
        matrix_element_list = [j[0] for j in result_list]
        R_0_list = [j[1] for j in result_list]; R_2_list = [j[2] for j in result_list]
        delta_0_list = [j[3] for j in result_list]; delta_2_list = [j[4] for j in result_list]
        
        plt.plot(hn_range, matrix_element_list)
        plt.yscale('log')
        plt.xlabel('photon energy (eV)')
        plt.ylabel('matrix element')
        plt.title(''.join(['matrix element for Z=', str(self.Z), ', l=', str(self.l)]))
        
        np.savetxt(save_file, np.array([hn_range, matrix_element_list, R_0_list, R_2_list,\
                                                   delta_0_list, delta_2_list]).T, delimiter=',')
        
        return (hn_range, matrix_element_list, R_0_list, R_2_list, delta_0_list, delta_2_list)
    
    def matrix_element_in_plane_dependence(self, hn, k_x_range, k_y_range, mode = 0,theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2,\
                                          save_file = 'matrix_element.csv'):
        '''
        calculate in-plane momentum dependence of cross-section
        params:
            hn: photon energy, in units of eV
            k_x_range, k_y_range: list of k-points along k_x and k_y. 
                                  if mode == 0:
                                      a single cut is generated
                                  if mode == 1:
                                      A meshgrid is generated and cross-section is calculated at each k-point on meshgrid
            mode: control scheme for k_pts sampling
            theta_epsilon, phi_epsilon: same as the definition in function matrix_element_core
            save_file: the saved file name
            
        return:
            (k_pts, matrix_element_list)
            k_pts: a list of k-points
            matrix_element_list: the corresponding list of cross-section
        '''
        if mode == 0:
            k_pts = [[k_x_range[j], k_y_range[j]] for j in range(len(k_x_range))]
        if mode == 1:
            k_pts = [[x,y] for x in k_x_range for y in k_y_range]
        result_list = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.matrix_element_core)(hn, theta_epsilon, phi_epsilon, j[0], j[1]) for j in k_pts)
        matrix_element_list = [j[0] for j in result_list]        
        np.savetxt(save_file, np.array([[j[0] for j in k_pts], [j[1] for j in k_pts], matrix_element_list]).T, delimiter=',')
        return (k_pts, matrix_element_list)
    
    def total_cross_section_core(self, hn, N_nl):
        '''
        Calculate the total cross-section at specific photon energy for random-oriented ensemble of atoms
        params:
            hn: photon energy, in units of eV
            N_nl: the number of electrons in the nl subshell (=2(2l+1) if filled)
            
        return:
            total cross section
        '''
        print('--- calculating total cross section at hn = ', hn, ' eV ---')
        if self.l == 0:
            R_1, delta_1 = self._radial_int_and_delta(hn, self.l, self.l+1)
            return self.prefactor*(N_nl)*hn*(self.l+1)*np.power(R_1, 2)/(2*self.l+1)
        
        if self.l == 1:
            R_0, delta_0 = self._radial_int_and_delta(hn, self.l, self.l-1)
            R_2, delta_2 = self._radial_int_and_delta(hn, self.l, self.l+1)
            return self.prefactor*(N_nl)*hn*(self.l*np.power(R_0,2) + (self.l+1)*np.power(R_2, 2))/(2*self.l+1)
        
        if self.l == 2:
            R_1, delta_1 = self._radial_int_and_delta(hn, self.l, self.l-1)
            R_3, delta_3 = self._radial_int_and_delta(hn, self.l, self.l+1)
            return self.prefactor*(N_nl)*hn*(self.l*np.power(R_1,2) + (self.l+1)*np.power(R_3, 2))/(2*self.l+1)
        
    def total_cross_section_hn_dependence(self, hn_range, N_nl, save_file = 'total_cross_section.csv'):
        '''
        Calculate the photon energy dependence of the cross-section
        params:
            hn_range: a list of photon-energy to calculate
            N_nl: the number of electrons in the nl subshell (=2(2l+1) if filled)
            
        return:
            a two-dimensional array, with first column the photon energy and second column the total cross section
        '''
        result = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.total_cross_section_core)(j, N_nl) for j in hn_range)
        
        plt.plot(hn_range, result)
        plt.ylabel('total cross section')
        plt.yscale('log')
        plt.xlabel('photon energy (eV)')
        
        np.savetxt(save_file, np.array([hn_range, result]).T, delimiter=',')
        
        return np.array([hn_range, result]).T
        
    def _radial_int_and_delta(self, hn, l_bound, l_free):
        '''
        Calculate the radial integral R_l(E_{kin}).
        params:
            hn: photon energy, in units of eV
            l_bound: angular momentum of bound state
            l_free: angular momentum of free state
        
        return:
            (R_l(E_{kin}), delta)
            R_l(E_{kin}): the integral of radial wave function
            delta: the phase shift of the continuum final state
        '''
        radial_wave = radial_schrodinger_wave({'Z': self.Z, 'l': l_bound})
        R_1 = radial_wave.P_func_and_delta(self.E_nl)

        radial_wave = radial_schrodinger_wave({'Z': self.Z, 'l': l_free})
        R_2, delta = radial_wave.P_func_and_delta(hn/13.605693 + self.E_nl)
        
        def integrand(x):
            return np.interp(x, radial_wave.r_mesh, [R_1[j]*R_2[j]*radial_wave.r_mesh[j] for j in range(len(radial_wave.r_mesh))])
        
        return (integrate.quad(integrand, self.r_mesh_start, self.r_mesh_end, limit=10000)[0], delta)
    
    def _momentum2angle(self, hn, k_x, k_y):
        '''
        Convert momentum (k_x, k_y) to photoemission angle
        params:
            hn: photon energy, in units of eV
            k_x, k_y: in-plane momentum, in units of 1/Angstrom
            
        return:
            theta_k, phi_k: corresponding photoemission angle, in units of rad
        '''
        
        if hn + self.E_nl*13.605693 < 0:
            print('Photoemission prohibited at hn=', hn, ' eV, k_x=', k_x, ', ky=', k_y, ' not allowed.')
            return None
        else:
            k = 0.512*np.sqrt(hn + self.E_nl*13.605693)
            if (k_x**2+k_y**2 >= k**2):
                print('Photoemission prohibited at hn=', hn, ' eV, k_x=', k_x, ', ky=', k_y, ' not allowed.')
                return None
            else:
                theta_k = np.arctan(np.sqrt(k_x**2+k_y**2)/np.sqrt(k**2-k_x**2-k_y**2))
                if k_x<0:
                    phi_k = np.pi+np.arctan(k_y/k_x)
                else:
                    phi_k = np.arctan(k_y/k_x)
                return (theta_k, phi_k)            


config = {'Z': 34,
         'orbital': '3pz'}

class matrix_element_scenario_3:
    '''
    Calculate atomic photoemission matrix element under dipole approximation with
    initial state: bound state under Hartree-Fock central potential
    final state: scattering state under free-electron final state
    '''
    def __init__(self, config):
        '''
        Initialize class
        params:
            config: dict, class configuration, with following keys:
                - config["Z"]: int, the nuclear charge of the atom
                - config["orbital"]: str, the conventional name of orbital, eg. "1s", "2px", "2py", "2pz",\
                                    "3dxy", "3dxz", "3dyz", "3dz2", "3dx2-y2", etc.
        '''
        self.config = config
        self.Z = config['Z']
        self.orbital = config['orbital']
        
        HF_potential = h5py.File('HF_potential.hdf5','r')
        self.U_data = np.array([list(HF_potential[str(self.Z)]['x'][:]), list(HF_potential[str(self.Z)]['U'][:])]).T
        self.U_data[:,0] *=  0.88534138*np.power(self.Z, -1/3)
        try:
            orbital_idx = [j for j in range(len(HF_potential[str(self.Z)]['orbital'][:])) if HF_potential[str(self.Z)]['orbital'][j].decode('utf-8') == self.orbital[0:2]][0]
        except:
            orbital_idx = [j for j in range(len(HF_potential[str(self.Z)]['orbital'][:])) if HF_potential[str(self.Z)]['orbital'][j] == self.orbital[0:2]][0]
        self.E_nl = HF_potential[str(self.Z)]['energy'][orbital_idx]
        HF_potential.close()

        self.n = int(self.orbital[0])
        if self.orbital[1] == 's':
            self.l = 0
        if self.orbital[1] == 'p':
            self.l = 1
        if self.orbital[1] == 'd':
            self.l = 2
        
        self.r_mesh_start = self.U_data[0,0]
        self.r_mesh_end = 100*self.U_data[-1,0]
        self.r_mesh_delta = 0.001
        self.prefactor = 4*np.pi*np.power(0.52918,2)/(3*13704)
        
    def matrix_element_core(self, hn, theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2, k_x=0.01, k_y=0.01):
        '''
        Calculate matrix element at given photon energy and in-plane momentum.
        params:
            hn: float, photon energy, in the unit of eV
            theta_epsilon, phi_epsilon: float, in the unit of rad. \
                                        The direction of electric vector of incident light.
                                        At Stanford Syncrotron Radiation Lightsource Beamline 5-2,
                                        LH (p) polarization: (theta_epsilon, phi_epsilon)=2*np.pi/9, np.pi/2
                                        LV (s) polarization: (theta_epsilon, phi_epsilon)=np.pi/2, 0
            k_x, k_y: float, the in-plane momentum, in the units of Angstrom. The corresponding photoemission angle are calculated correspondingly.
            
        return:
            (cross_section, R_0, R_2)
            where:
                cross_section: the matrix element 
                R_0, R_2: the radial integral of the two l'=l\pm 1 channels
        '''
        print('--- calculating matrix element at hn = ', hn, ' eV, k_x=', k_x, ', k_y=', k_y, ' ---')
        theta_k, phi_k = self._momentum2angle(hn, k_x, k_y)
        epsilon_x = np.sin(theta_epsilon)*np.cos(phi_epsilon)
        epsilon_y = np.sin(theta_epsilon)*np.sin(phi_epsilon)
        epsilon_z = np.cos(theta_epsilon)
        
        if self.l == 0:
            R_1 = self._radial_int(hn, self.l, self.l+1)
            k = 0.512*np.sqrt(hn + self.E_nl*13.605693)
            k_z = np.sqrt(k**2-k_x**2-k_y**2)
            return (self.prefactor*hn*4*np.pi*np.power(R_1, 2)*np.power(epsilon_x*k_x+epsilon_y*k_y+epsilon_z*k_z, 2), R_1, None)
        
        if self.l == 1:
            R_0 = self._radial_int(hn, self.l, self.l-1)
            R_2 = self._radial_int(hn, self.l, self.l+1)
            
            if self.orbital[1:] == 'px':
                X_0 = 2*np.sqrt(2*np.pi)*R_0*np.sqrt(1/3)*epsilon_x
                X_2 = 2*np.sqrt(np.pi)*R_2*(-np.sqrt(3/4)*np.power(np.sin(theta_k), 2)*(epsilon_x*np.cos(2*phi_k)\
                      +epsilon_y*np.sin(2*phi_k))+epsilon_x*np.sqrt(1/12)*(3*np.power(np.cos(theta_k),2)-1)\
                      -epsilon_z*np.sqrt(3)*np.sin(theta_k)*np.cos(theta_k)*np.cos(phi_k))
                
            if self.orbital[1:] == 'py':
                X_0 = 2*np.sqrt(np.pi)*R_0*np.sqrt(1/3)*epsilon_y
                X_2 = 2*np.sqrt(np.pi)*R_2*(np.sqrt(3/4)*np.power(np.sin(theta_k),2)*(-epsilon_x*np.sin(2*phi_k)\
                        +epsilon_y*np.cos(2*phi_k))+epsilon_y*np.sqrt(1/12)*(3*np.power(np.cos(theta_k),2)-1)\
                        -epsilon_z*np.sqrt(3)*np.sin(theta_k)*np.cos(theta_k)*np.sin(phi_k))
            
            if self.orbital[1:] == 'pz':
                X_0 = 2*np.sqrt(2*np.pi)*R_0*np.sqrt(1/6)*epsilon_z
                X_2 = 2*np.sqrt(2*np.pi)*R_2*(np.sqrt(3/2)*np.sin(theta_k)*np.cos(theta_k)\
                        *(-epsilon_x*np.cos(phi_k)-epsilon_y*np.sin(phi_k))\
                        - epsilon_z*np.sqrt(1/6)*(3*np.power(np.cos(theta_k),2)-1))
            
            return (self.prefactor*hn*np.power(X_0+X_2, 2), R_0, R_2)
        
        if self.l == 2:
            R_1 = self._radial_int(hn, self.l, self.l-1)
            R_3 = self._radial_int(hn, self.l, self.l+1)
            
            if self.orbital[1:] == 'dxy':
                X_1 = 2*np.sqrt(np.pi)*R_1*np.sqrt(3/5)*np.sin(theta_k)*(epsilon_x*np.sin(phi_k)+epsilon_y*np.cos(phi_k))
                X_3 = -2*np.sqrt(np.pi)*R_3*(-np.sqrt(3/80)*np.sin(phi_k)*(5*np.power(np.cos(theta_k),2)-1)*(epsilon_x*np.sin(phi_k)\
                        + epsilon_y*np.cos(phi_k))+np.sqrt(15/16)*np.power(np.sin(theta_k),3)*(epsilon_x*np.sin(3*phi_k)-epsilon_y*np.cos(3*phi_k))\
                        + epsilon_z*np.sqrt(15/4)*np.power(np.sin(theta_k),2)*np.cos(theta_k)*np.sin(2*phi_k))
                
            if self.orbital[1:] == 'dxz':
                X_1 = 2*np.sqrt(np.pi)*R_1*(np.sqrt(3/5)*(epsilon_x*np.cos(theta_k)+epsilon_z*np.sin(theta_k)*np.cos(phi_k)))
                X_3 = -2*np.sqrt(np.pi)*R_3*(np.sqrt(15/4)*np.power(np.sin(theta_k),2)*np.cos(theta_k)*(epsilon_x*np.cos(2*phi_k)\
                    + epsilon_y*np.sin(2*phi_k))-epsilon_x*np.sqrt(3/20)*(5*np.power(np.cos(theta_k),3)-3*np.cos(theta_k))\
                    + epsilon_z*np.sqrt(3/5)*np.sin(theta_k)*(5*np.power(np.cos(theta_k),2)-1)*np.cos(phi_k))
                
            if self.orbital[1:] == 'dyz':
                X_1 = 2*np.sqrt(np.pi)*R_1*(np.sqrt(3/5)*(epsilon_y*np.cos(theta_k)+epsilon_z*np.sin(theta_k)*np.sin(phi_k)))
                X_3 = -2*np.sqrt(np.pi)*R_3*(np.sqrt(15/4)*np.power(np.sin(theta_k),2)*np.cos(theta_k)*(epsilon_x*np.sin(2*phi_k)\
                    - epsilon_y*np.cos(2*phi_k))-epsilon_y*np.sqrt(3/20)*(5*np.power(np.cos(theta_k),3)-3*np.cos(theta_k))\
                    + epsilon_z*np.sqrt(3/5)*np.sin(theta_k)*(5*np.power(np.cos(theta_k),2)-1)*np.sin(phi_k))
                
            if self.orbital[1:] == 'dx2-y2':
                X_1 = -2*np.sqrt(np.pi)*R_1*np.sqrt(3/5)*np.sin(theta_k)*(-epsilon_x*np.cos(phi_k)+epsilon_y*np.sin(phi_k))
                X_3 = -2*np.sqrt(np.pi)*R_3*(np.sqrt(3/80)*np.sin(theta_k)*(5*np.power(np.cos(theta_k),2)-1)*(-epsilon_x*np.cos(phi_k)+epsilon_y*np.sin(phi_k))\
                                            +np.sqrt(15/16)*np.power(np.sin(theta_k),3)*(epsilon_x*np.cos(3*phi_k)+epsilon_y*np.sin(3*phi_k))\
                                            +epsilon_z*np.sqrt(15/4)*np.power(np.sin(theta_k),2)*np.cos(theta_k)*np.cos(2*phi_k))
                
            if self.orbital[1:] == 'dz2':
                X_1 = -2*np.sqrt(2*np.pi)*R_1*(np.sqrt(1/10)*np.sin(theta_k)*(epsilon_x*np.cos(phi_k)+epsilon_y*np.sin(phi_k))\
                                               -epsilon_z*np.sqrt(2/5)*np.cos(theta_k))
                X_3 = -2*np.sqrt(2*np.pi)*R_3*(np.sqrt(9/40)*np.sin(theta_k)*(5*np.power(np.cos(theta_k),2)-1)*(epsilon_x*np.cos(phi_k)+epsilon_y*np.sin(phi_k))\
                                              +epsilon_z*np.sqrt(9/40)*(5*np.power(np.cos(theta_k),3)-3*np.cos(theta_k)))
            
            return (self.prefactor*hn*np.power(X_1+X_3, 2), R_1, R_3)
    
    def matrix_element_hn_dependence(self, hn_range, \
                                     theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2, k_x=0.01, k_y=0.01,\
                                    save_file = 'matrix_element.csv'):
        '''
        Calculate photon-energy dependence of cross-section
        params:
            hn_range: list, list of photon energy to calculate
            theta_epsilon, phi_epsilon, k_x, k_y: same as the definition in function self.matrix_element_core
            save_file: the saved file name
            
        return:
            (hn_range, matrix_element_list, R_0_list, R_2_list)
            hn_range: a list of photon energy to calculate
            matrix_element_list: the corresponding list of cross-section
            R_0_list, R_2_list: the corresponding list of R_0, R_2, delta_0, delta_2, same as the definition in function matrix_element_core
        '''

        result_list = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.matrix_element_core)(hn, theta_epsilon, phi_epsilon, k_x, k_y) for hn in hn_range)
        matrix_element_list = [j[0] for j in result_list]
        R_0_list = [j[1] for j in result_list]; R_2_list = [j[2] for j in result_list]
        
        plt.plot(hn_range, matrix_element_list)
        plt.yscale('log')
        plt.xlabel('photon energy (eV)')
        plt.ylabel('matrix element')
        plt.title(''.join(['matrix element for Z=', str(self.Z), ', l=', str(self.l)]))
        
        np.savetxt(save_file, np.array([hn_range.tolist(), matrix_element_list, R_0_list, R_2_list]).T, delimiter=',')
        
        return (hn_range, matrix_element_list, R_0_list, R_2_list)
    
    def matrix_element_in_plane_dependence(self, hn, k_x_range, k_y_range, mode=1, theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2,\
                                          save_file = 'matrix_element.csv'):
        '''
        calculate in-plane momentum dependence of cross-section
        params:
            hn: photon energy, in units of eV
            k_x_range, k_y_range: list of k-points along k_x and k_y. 
                                  if mode == 0:
                                      a single cut is generated
                                  if mode == 1:
                                      A meshgrid is generated and cross-section is calculated at each k-point on meshgrid
            mode: control scheme for k_pts sampling
            theta_epsilon, phi_epsilon: same as the definition in function matrix_element_core
            save_file: the saved file name
            
        return:
            (k_pts, matrix_element_list)
            k_pts: a list of k-points
            matrix_element_list: the corresponding list of cross-section
        '''
        if mode == 0:
            k_pts = [[k_x_range[j], k_y_range[j]] for j in range(len(k_x_range))]
        if mode == 1:
            k_pts = [[x,y] for x in k_x_range for y in k_y_range]
        result_list = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.matrix_element_core)(hn, theta_epsilon, phi_epsilon, j[0], j[1]) for j in k_pts)
        matrix_element_list = [j[0] for j in result_list]        
        np.savetxt(save_file, np.array([[j[0] for j in k_pts], [j[1] for j in k_pts], matrix_element_list]).T, delimiter=',')
        
        return (k_pts, matrix_element_list)

    
    def total_cross_section_core(self, hn, N_nl):
        '''
        Calculate the total cross-section at specific photon energy for random-oriented ensemble of atoms
        params:
            hn: photon energy, in units of eV
            N_nl: the number of electrons in the nl subshell (=2(2l+1) if filled)
            
        return:
            total cross section
        '''
        print('--- calculating total cross section at hn = ', hn, ' eV ---')
        if self.l == 0:
            R_1, delta_1 = self._radial_int_and_delta(hn, self.l, self.l+1)
            return self.prefactor*(N_nl)*hn*(self.l+1)*np.power(R_1, 2)/(2*self.l+1)
        
        if self.l == 1:
            R_0, delta_0 = self._radial_int_and_delta(hn, self.l, self.l-1)
            R_2, delta_2 = self._radial_int_and_delta(hn, self.l, self.l+1)
            return self.prefactor*(N_nl)*hn*(self.l*np.power(R_0,2) + (self.l+1)*np.power(R_2, 2))/(2*self.l+1)
        
        if self.l == 2:
            R_1, delta_1 = self._radial_int_and_delta(hn, self.l, self.l-1)
            R_3, delta_3 = self._radial_int_and_delta(hn, self.l, self.l+1)
            return self.prefactor*(N_nl)*hn*(self.l*np.power(R_1,2) + (self.l+1)*np.power(R_3, 2))/(2*self.l+1)
        
    def total_cross_section_hn_dependence(self, hn_start, hn_end, hn_delta, N_nl, save_file = 'total_cross_section.csv'):
        '''
        Calculate the photon energy dependence of the cross-section
        params:
            hn_range: a list of photon-energy to calculate
            N_nl: the number of electrons in the nl subshell (=2(2l+1) if filled)
            
        return:
            a two-dimensional array, with first column the photon energy and second column the total cross section
        '''
        hn_range = np.arange(hn_start, hn_end, hn_delta)
        result = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.total_cross_section_core)(j, N_nl) for j in hn_range)
        
        plt.plot(hn_range, result)
        plt.title('total cross section')
        plt.yscale('log')
        
        np.savetxt(save_file, np.array([hn_range.tolist(), result]).T, delimiter=',')
        
        return np.array([hn_range.tolist(), result]).T
    
    def _radial_int(self, hn, l_bound, l_free):
        '''
        calculate R_l(E_{kin}).
        params:
            hn: photon energy, in units of eV
            l_bound: angular momentum of bound state
            l_free: angular momentum of free state
            
        '''
        Bohr_radius = 0.5292 # in the unit of Angstrom
        k_f = 0.512*np.sqrt(hn + self.E_nl*13.605693) # in the unit of 1/Angstrom
        
        radial_wave = radial_schrodinger_wave({'Z': self.Z, 'l': l_bound})
        R_1 = radial_wave.P_func_and_delta(self.E_nl)
        
        def final_state_radial(r):
            return scipy.special.spherical_jn(l_free, k_f*r)
        
        def integrand(x):
            return np.interp(x, radial_wave.r_mesh, [R_1[j]*final_state_radial(radial_wave.r_mesh[j]*Bohr_radius)*np.power(radial_wave.r_mesh[j],2) for j in range(len(radial_wave.r_mesh))])
        
        return integrate.quad(integrand, self.r_mesh_start, self.r_mesh_end, limit=10000)[0]
    
    def _momentum2angle(self, hn, k_x, k_y):
        '''
        convert momentum (k_x, k_y) to photoemission angle
        params:
            hn: photon energy, in units of eV
            k_x, k_y: in-plane momentum, in units of 1/Angstrom
            
        return:
            theta_k, phi_k: corresponding photoemission angle, in units of rad
        '''
        alpha = 0.512
        
        if hn + self.E_nl*13.605693 < 0:
            print('Photoemission prohibited at hn=', hn, ' eV, k_x=', k_x, ', ky=', k_y, ' not allowed.')
            return None
        else:
            k = alpha*np.sqrt(hn + self.E_nl*13.605693)
            if (k_x**2+k_y**2 >= k**2):
                print('Photoemission prohibited at hn=', hn, ' eV, k_x=', k_x, ', ky=', k_y, ' not allowed.')
                return None
            else:
                theta_k = np.arctan(np.sqrt(k_x**2+k_y**2)/np.sqrt(k**2-k_x**2-k_y**2))
                if k_x<0:
                    phi_k = np.pi+np.arctan(k_y/k_x)
                else:
                    phi_k = np.arctan(k_y/k_x)
                return (theta_k, phi_k)            


config = {'Z': 34,
          'Z_eff': 6.95, # effective nuclear charge, determined by the slater's rule
         'orbital': '3pz'}

class matrix_element_scenario_2:
    '''
    Calculate atomic photoemission matrix element under dipole approximation with
    initial state: hydrogen-like bound state
    final state: free-electron final state
    '''
    def __init__(self, config):
        '''
        Initialize class
        params:
            config: dict, class configuration, with following keys:
                - config["Z"]: int, the nuclear charge of the atom
                - config["Z_eff"]: float, the effective nuclear charge determined by Slater's rule
                - config["orbital"]: str, the conventional name of orbital, eg. "1s", "2px", "2py", "2pz",\
                                    "3dxy", "3dxz", "3dyz", "3dz2", "3dx2-y2", etc.
        '''
        self.config = config
        self.Z = config['Z']
        self.Z_eff = config['Z_eff']
        self.orbital = config['orbital']
        
        HF_potential = h5py.File('HF_potential.hdf5','r')
        try:
            orbital_idx = [j for j in range(len(HF_potential[str(self.Z)]['orbital'][:])) if HF_potential[str(self.Z)]['orbital'][j].decode('utf-8') == self.orbital[0:2]][0]
        except:
            orbital_idx = [j for j in range(len(HF_potential[str(self.Z)]['orbital'][:])) if HF_potential[str(self.Z)]['orbital'][j] == self.orbital[0:2]][0]
        HF_potential.close()
        
        self.E_nl = HF_potential[str(self.Z)]['energy'][orbital_idx]        
        self.n = int(self.orbital[0])
        if self.orbital[1] == 's':
            self.l = 0
        if self.orbital[1] == 'p':
            self.l = 1
        if self.orbital[1] == 'd':
            self.l = 2
        
        self.prefactor = 4*np.pi*np.power(0.52918,2)/(3*13704)
        
    def matrix_element_core(self, hn, theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2, k_x=0.01, k_y=0.01):
        '''
        Calculate matrix element at given photon energy and in-plane momentum.
        params:
            hn: float, photon energy, in the unit of eV
            theta_epsilon, phi_epsilon: float, in the unit of rad. \
                                        The direction of electric vector of incident light.
                                        At Stanford Syncrotron Radiation Lightsource Beamline 5-2,
                                        LH (p) polarization: (theta_epsilon, phi_epsilon)=2*np.pi/9, np.pi/2
                                        LV (s) polarization: (theta_epsilon, phi_epsilon)=np.pi/2, 0
            k_x, k_y: float, the in-plane momentum, in the units of Angstrom. The corresponding photoemission angle are calculated correspondingly.
            
        return:
            (cross_section, R_0, R_2, delta_0, delta_2)
            where:
                cross_section: the matrix element 
                R_0, R_2: the radial integral of the two l'=l\pm 1 channels
        '''
        print('--- calculating matrix element at hn = ', hn, ' eV, k_x=', k_x, ', k_y=', k_y, ' ---')
        theta_k, phi_k = self._momentum2angle(hn, k_x, k_y)
        epsilon_x = np.sin(theta_epsilon)*np.cos(phi_epsilon)
        epsilon_y = np.sin(theta_epsilon)*np.sin(phi_epsilon)
        epsilon_z = np.cos(theta_epsilon)
        
        if self.l == 0:
            R_1 = self._radial_int(hn, self.l, self.l+1)
            k = 0.512*np.sqrt(hn + self.E_nl*13.605693)
            k_z = np.sqrt(k**2-k_x**2-k_y**2)
            return (self.prefactor*hn*4*np.pi*np.power(R_1, 2)*np.power(epsilon_x*k_x+epsilon_y*k_y+epsilon_z*k_z, 2), R_1, None)
        
        if self.l == 1:
            R_0 = self._radial_int(hn, self.l, self.l-1)
            R_2 = self._radial_int(hn, self.l, self.l+1)
            
            if self.orbital[1:] == 'px':
                X_0 = 2*np.sqrt(2*np.pi)*R_0*np.sqrt(1/3)*epsilon_x
                X_2 = 2*np.sqrt(np.pi)*R_2*(-np.sqrt(3/4)*np.power(np.sin(theta_k), 2)*(epsilon_x*np.cos(2*phi_k)\
                      +epsilon_y*np.sin(2*phi_k))+epsilon_x*np.sqrt(1/12)*(3*np.power(np.cos(theta_k),2)-1)\
                      -epsilon_z*np.sqrt(3)*np.sin(theta_k)*np.cos(theta_k)*np.cos(phi_k))
                
            if self.orbital[1:] == 'py':
                X_0 = 2*np.sqrt(np.pi)*R_0*np.sqrt(1/3)*epsilon_y
                X_2 = 2*np.sqrt(np.pi)*R_2*(np.sqrt(3/4)*np.power(np.sin(theta_k),2)*(-epsilon_x*np.sin(2*phi_k)\
                        +epsilon_y*np.cos(2*phi_k))+epsilon_y*np.sqrt(1/12)*(3*np.power(np.cos(theta_k),2)-1)\
                        -epsilon_z*np.sqrt(3)*np.sin(theta_k)*np.cos(theta_k)*np.sin(phi_k))
            
            if self.orbital[1:] == 'pz':
                X_0 = 2*np.sqrt(2*np.pi)*R_0*np.sqrt(1/6)*epsilon_z
                X_2 = 2*np.sqrt(2*np.pi)*R_2*(np.sqrt(3/2)*np.sin(theta_k)*np.cos(theta_k)\
                        *(-epsilon_x*np.cos(phi_k)-epsilon_y*np.sin(phi_k))\
                        - epsilon_z*np.sqrt(1/6)*(3*np.power(np.cos(theta_k),2)-1))
            
            return (self.prefactor*hn*np.power(X_0+X_2, 2), R_0, R_2)
        
        if self.l == 2:
            R_1 = self._radial_int(hn, self.l, self.l-1)
            R_3 = self._radial_int(hn, self.l, self.l+1)
            
            if self.orbital[1:] == 'dxy':
                X_1 = 2*np.sqrt(np.pi)*R_1*np.sqrt(3/5)*np.sin(theta_k)*(epsilon_x*np.sin(phi_k)+epsilon_y*np.cos(phi_k))
                X_3 = -2*np.sqrt(np.pi)*R_3*(-np.sqrt(3/80)*np.sin(phi_k)*(5*np.power(np.cos(theta_k),2)-1)*(epsilon_x*np.sin(phi_k)\
                        + epsilon_y*np.cos(phi_k))+np.sqrt(15/16)*np.power(np.sin(theta_k),3)*(epsilon_x*np.sin(3*phi_k)-epsilon_y*np.cos(3*phi_k))\
                        + epsilon_z*np.sqrt(15/4)*np.power(np.sin(theta_k),2)*np.cos(theta_k)*np.sin(2*phi_k))
                
            if self.orbital[1:] == 'dxz':
                X_1 = 2*np.sqrt(np.pi)*R_1*(np.sqrt(3/5)*(epsilon_x*np.cos(theta_k)+epsilon_z*np.sin(theta_k)*np.cos(phi_k)))
                X_3 = -2*np.sqrt(np.pi)*R_3*(np.sqrt(15/4)*np.power(np.sin(theta_k),2)*np.cos(theta_k)*(epsilon_x*np.cos(2*phi_k)\
                    + epsilon_y*np.sin(2*phi_k))-epsilon_x*np.sqrt(3/20)*(5*np.power(np.cos(theta_k),3)-3*np.cos(theta_k))\
                    + epsilon_z*np.sqrt(3/5)*np.sin(theta_k)*(5*np.power(np.cos(theta_k),2)-1)*np.cos(phi_k))
                
            if self.orbital[1:] == 'dyz':
                X_1 = 2*np.sqrt(np.pi)*R_1*(np.sqrt(3/5)*(epsilon_y*np.cos(theta_k)+epsilon_z*np.sin(theta_k)*np.sin(phi_k)))
                X_3 = -2*np.sqrt(np.pi)*R_3*(np.sqrt(15/4)*np.power(np.sin(theta_k),2)*np.cos(theta_k)*(epsilon_x*np.sin(2*phi_k)\
                    - epsilon_y*np.cos(2*phi_k))-epsilon_y*np.sqrt(3/20)*(5*np.power(np.cos(theta_k),3)-3*np.cos(theta_k))\
                    + epsilon_z*np.sqrt(3/5)*np.sin(theta_k)*(5*np.power(np.cos(theta_k),2)-1)*np.sin(phi_k))
                
            if self.orbital[1:] == 'dx2-y2':
                X_1 = -2*np.sqrt(np.pi)*R_1*np.sqrt(3/5)*np.sin(theta_k)*(-epsilon_x*np.cos(phi_k)+epsilon_y*np.sin(phi_k))
                X_3 = -2*np.sqrt(np.pi)*R_3*(np.sqrt(3/80)*np.sin(theta_k)*(5*np.power(np.cos(theta_k),2)-1)*(-epsilon_x*np.cos(phi_k)+epsilon_y*np.sin(phi_k))\
                                            +np.sqrt(15/16)*np.power(np.sin(theta_k),3)*(epsilon_x*np.cos(3*phi_k)+epsilon_y*np.sin(3*phi_k))\
                                            +epsilon_z*np.sqrt(15/4)*np.power(np.sin(theta_k),2)*np.cos(theta_k)*np.cos(2*phi_k))
                
            if self.orbital[1:] == 'dz2':
                X_1 = -2*np.sqrt(2*np.pi)*R_1*(np.sqrt(1/10)*np.sin(theta_k)*(epsilon_x*np.cos(phi_k)+epsilon_y*np.sin(phi_k))\
                                               -epsilon_z*np.sqrt(2/5)*np.cos(theta_k))
                X_3 = -2*np.sqrt(2*np.pi)*R_3*(np.sqrt(9/40)*np.sin(theta_k)*(5*np.power(np.cos(theta_k),2)-1)*(epsilon_x*np.cos(phi_k)+epsilon_y*np.sin(phi_k))\
                                              +epsilon_z*np.sqrt(9/40)*(5*np.power(np.cos(theta_k),3)-3*np.cos(theta_k)))
            
            return (self.prefactor*hn*np.power(X_1+X_3, 2), R_1, R_3)
    
    def matrix_element_hn_dependence(self, hn_range, \
                                     theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2, k_x=0.01, k_y=0.01,\
                                    save_file = 'matrix_element.csv'):
        '''
        Calculate photon-energy dependence of cross-section
        params:
            hn_range: list, list of photon energy to calculate
            theta_epsilon, phi_epsilon, k_x, k_y: same as the definition in function self.matrix_element_core
            save_file: the saved file name
            
        return:
            (hn_range, matrix_element_list, R_0_list, R_2_list)
            hn_range: a list of photon energy to calculate
            matrix_element_list: the corresponding list of cross-section
            R_0_list, R_2_list: the corresponding list of R_0, R_2, same as the definition in function matrix_element_core
        '''
        result_list = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.matrix_element_core)(hn, theta_epsilon, phi_epsilon, k_x, k_y) for hn in hn_range)
        matrix_element_list = [j[0] for j in result_list]
        R_0_list = [j[1] for j in result_list]; R_2_list = [j[2] for j in result_list]
        
        plt.plot(hn_range, matrix_element_list)
        plt.yscale('log')
        plt.xlabel('photon energy (eV)')
        plt.ylabel('matrix element')
        plt.title(''.join(['matrix element for Z=', str(self.Z), ', l=', str(self.l)]))
        
        np.savetxt(save_file, np.array([hn_range.tolist(), matrix_element_list, R_0_list, R_2_list]).T, delimiter=',')
        
        return (hn_range, matrix_element_list, R_0_list, R_2_list)
    
    def matrix_element_in_plane_dependence(self, hn, k_x_range, k_y_range, mode=1, theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2,\
                                          save_file = 'matrix_element.csv'):
        '''
        calculate in-plane momentum dependence of cross-section
        params:
            hn: photon energy, in units of eV
            k_x_range, k_y_range: list of k-points along k_x and k_y. 
                                  if mode == 0:
                                      a single cut is generated
                                  if mode == 1:
                                      A meshgrid is generated and cross-section is calculated at each k-point on meshgrid
            mode: control scheme for k_pts sampling
            theta_epsilon, phi_epsilon: same as the definition in function matrix_element_core
            save_file: the saved file name

        return:
            (k_pts, matrix_element_list)
            k_pts: a list of k-points
            matrix_element_list: the corresponding list of cross-section
        '''
        if mode == 0:
            k_pts = [[k_x_range[j], k_y_range[j]] for j in range(len(k_x_range))]
        if mode == 1:
            k_pts = [[x,y] for x in k_x_range for y in k_y_range]
        result_list = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.matrix_element_core)(hn, theta_epsilon, phi_epsilon, j[0], j[1]) for j in k_pts)
        matrix_element_list = [j[0] for j in result_list]        
        np.savetxt(save_file, np.array([[j[0] for j in k_pts], [j[1] for j in k_pts], matrix_element_list]).T, delimiter=',')
        
        return (k_pts, matrix_element_list)

    def total_cross_section_core(self, hn, N_nl):
        '''
        Calculate the total cross section for an randomly-oriented ensemble of atoms
        params:
            hn: photon energy, in units of eV
            N_nl: the number of electrons in the nl subshell (=2(2l+1) if filled)
            
        return:
            total cross section 
        '''
        print('--- calculating total cross section at hn = ', hn, ' eV ---')
        if self.l == 0:
            R_1, delta_1 = self._radial_int_and_delta(hn, self.l, self.l+1)
            return self.prefactor*(N_nl)*hn*(self.l+1)*np.power(R_1, 2)/(2*self.l+1)
        
        if self.l == 1:
            R_0, delta_0 = self._radial_int_and_delta(hn, self.l, self.l-1)
            R_2, delta_2 = self._radial_int_and_delta(hn, self.l, self.l+1)
            return self.prefactor*(N_nl)*hn*(self.l*np.power(R_0,2) + (self.l+1)*np.power(R_2, 2))/(2*self.l+1)
        
        if self.l == 2:
            R_1, delta_1 = self._radial_int_and_delta(hn, self.l, self.l-1)
            R_3, delta_3 = self._radial_int_and_delta(hn, self.l, self.l+1)
            return self.prefactor*(N_nl)*hn*(self.l*np.power(R_1,2) + (self.l+1)*np.power(R_3, 2))/(2*self.l+1)
        
    def total_cross_section_hn_dependence(self, hn_start, hn_end, hn_delta, N_nl, save_file = 'total_cross_section.csv'):
        '''
        Calculate the photon energy dependence of the cross-section
        params:
            hn_range: a list of photon-energy to calculate
            N_nl: the number of electrons in the nl subshell (=2(2l+1) if filled)
            
        return:
            a two-dimensional array, with first column the photon energy and second column the total cross section
       '''
        hn_range = np.arange(hn_start, hn_end, hn_delta)
        result = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.total_cross_section_core)(j, N_nl) for j in hn_range)
        
        plt.plot(hn_range, result)
        plt.title('total cross section')
        plt.yscale('log')
        
        np.savetxt(save_file, np.array([hn_range.tolist(), result]).T, delimiter=',')
        
        return np.array([hn_range.tolist(), result]).T
    
    def _radial_int(self, hn, l_bound, l_free):
        '''
        calculate R_l(E_{kin}).
        params:
            hn: photon energy, in units of eV
            l_bound: angular momentum of bound state
            l_free: angular momentum of free state
            
        '''
        Bohr_radius = 0.5292 # in the unit of Angstrom
        k_f = 0.512*np.sqrt(hn + self.E_nl*13.605693) # in the unit of 1/Angstrom
        
        def initial_state_radial(r):
            return np.sqrt(np.power(2/(self.n*Bohr_radius), 3)*np.math.factorial(self.n-l_bound-1)/(2*self.n*np.math.factorial(self.n+l_bound)))\
                    *np.exp(-self.Z_eff*r/(self.n*Bohr_radius))*np.power(2*self.Z_eff*r/(self.n*Bohr_radius), l_bound)*scipy.special.genlaguerre(self.n-l_bound-1,2*l_bound+1)(2*self.Z_eff*r/(self.n*Bohr_radius))
        
        def final_state_radial(r):
            return scipy.special.spherical_jn(l_free, k_f*r)
        
        def integrand(r):
            return initial_state_radial(r)*final_state_radial(r)*np.power(r,3)
        
        return integrate.quad(integrand, 0.001, 50*Bohr_radius, limit=10000)[0]
    
    def _momentum2angle(self, hn, k_x, k_y):
        '''
        convert momentum (k_x, k_y) to photoemission angle
        params:
            hn: photon energy, in units of eV
            k_x, k_y: in-plane momentum, in units of 1/Angstrom
            
        return:
            theta_k, phi_k: corresponding photoemission angle, in units of rad
        '''
        alpha = 0.512
        if hn + self.E_nl*13.605693 < 0:
            print('Photoemission prohibited at hn=', hn, ' eV, k_x=', k_x, ', ky=', k_y, ' not allowed.')
            return None
        else:
            k = alpha*np.sqrt(hn + self.E_nl*13.605693)
            if (k_x**2+k_y**2 >= k**2):
                print('Photoemission prohibited at hn=', hn, ' eV, k_x=', k_x, ', ky=', k_y, ' not allowed.')
                return None
            else:
                theta_k = np.arctan(np.sqrt(k_x**2+k_y**2)/np.sqrt(k**2-k_x**2-k_y**2))
                if k_x<0:
                    phi_k = np.pi+np.arctan(k_y/k_x)
                else:
                    phi_k = np.arctan(k_y/k_x)
                return (theta_k, phi_k)            

class matrix_element:
    def __init__(self, scenario, config):
        '''
        Initialize class
        params:
            scenario: int, 1 - scenario 1; 2 - scenario 2; 3 - scenario 3
            config: dict, class configuration, with following keys:
                if scenario == 1 or 3:
                    - config["Z"]: int, the nuclear charge of the atom
                    - config["orbital"]: str, the conventional name of orbital, eg. "1s", "2px", "2py", "2pz",\
                                        "3dxy", "3dxz", "3dyz", "3dz2", "3dx2-y2", etc.
                if scenario == 2:
                    - config["Z"]: int, the nuclear charge of the atom
                    - config["Z_eff"]: float, the effective nuclear charge determined by Slater's rule
                    - config["orbital"]: str, the conventional name of orbital, eg. "1s", "2px", "2py", "2pz",\
                                        "3dxy", "3dxz", "3dyz", "3dz2", "3dx2-y2", etc.
        '''
        if scenario == 1:
            self.matrix_element = matrix_element_scenario_1(config)
        if scenario == 2:
            self.matrix_element = matrix_element_scenario_2(config)
        if scenario == 3:
            self.matrix_element = matrix_element_scenario_3(config)
    
    def matrix_element_core(self, hn, theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2, k_x=0.01, k_y=0.01):
        '''
        Calculate matrix element at given photon energy and in-plane momentum.
        params:
            hn: float, photon energy, in the unit of eV
            theta_epsilon, phi_epsilon: float, in the unit of rad. \
                                        The direction of electric vector of incident light.
                                        At Stanford Syncrotron Radiation Lightsource Beamline 5-2,
                                        LH (p) polarization: (theta_epsilon, phi_epsilon)=2*np.pi/9, np.pi/2
                                        LV (s) polarization: (theta_epsilon, phi_epsilon)=np.pi/2, 0
            k_x, k_y: float, the in-plane momentum, in the units of Angstrom. The corresponding photoemission angle are calculated correspondingly.
            
        return:
            if scenario == 1:
                (cross_section, R_0, R_2, delta_0, delta_2)
                where:
                    cross_section: the matrix element 
                    R_0, R_2: the radial integral of the two l'=l\pm 1 channels
                    delta_0, delta_2: the phase shift of the initial state and final state
            if scenario == 2 or 3:
                (cross_section, R_0, R_2)
                where:
                    cross_section: the matrix element 
                    R_0, R_2: the radial integral of the two l'=l\pm 1 channels
        '''
        return self.matrix_element.matrix_element_core(hn, theta_epsilon, phi_epsilon, k_x, k_y)
    
    def matrix_element_hn_dependence(self, hn_range, \
                                     theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2, k_x=0.01, k_y=0.01,\
                                    save_file = 'matrix_element.csv'):
        '''
        Calculate photon-energy dependence of cross-section
        params:
            hn_range: list, list of photon energy to calculate
            theta_epsilon, phi_epsilon, k_x, k_y: same as the definition in function self.matrix_element_core
            save_file: the saved file name
            
        return:
            if scenario == 1:
                (hn_range, matrix_element_list, R_0_list, R_2_list, delta_0_list, delta_2_list)
                hn_range: a list of photon energy to calculate
                matrix_element_list: the corresponding list of cross-section
                R_0_list, R_2_list: the corresponding list of R_0, R_2, same as the definition in function matrix_element_core
                delta_0_list, delta_2_list: the corresponding list of delta_0, delta_2, same as the definition in function matrix_element_core
            if scenario == 2 or 3:
                (hn_range, matrix_element_list, R_0_list, R_2_list)
                hn_range: a list of photon energy to calculate
                matrix_element_list: the corresponding list of cross-section
                R_0_list, R_2_list: the corresponding list of R_0, R_2, same as the definition in function matrix_element_core
        '''
        return self.matrix_element.matrix_element_hn_dependence(hn_range, theta_epsilon, phi_epsilon, k_x, k_y, save_file)
    
    def total_cross_section_core(self, hn, N_nl):
        '''
        Calculate the total cross section for an randomly-oriented ensemble of atoms
        params:
            hn: photon energy, in units of eV
            N_nl: the number of electrons in the nl subshell (=2(2l+1) if filled)
            
        return:
            total_cross_section
        '''
        return self.matrix_element.total_cross_section_core(hn, N_nl)
        
    def total_cross_section_hn_dependence(self, hn_start, hn_end, hn_delta, N_nl, save_file = 'total_cross_section.csv'):
        '''
        Calculate the photon energy dependence of the cross-section
        params:
            hn_range: a list of photon-energy to calculate
            N_nl: the number of electrons in the nl subshell (=2(2l+1) if filled)
            
        return:
            total_cross_section: np.array, with the first column photon energies and the second column total cross section
       '''
        return self.matrix_element.total_cross_section_hn_dependence(hn_start, hn_end, hn_delta, N_nl, save_file)
        
    def matrix_element_in_plane_dependence(self, hn, k_x_range, k_y_range, mode = 0, theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2,\
                                          save_file = 'matrix_element.csv'):
        '''
        calculate in-plane momentum dependence of cross-section
        params:
            hn: photon energy, in units of eV
            k_x_range, k_y_range: list of k-points along k_x and k_y. 
                                  if mode == 0:
                                      a single cut is generated
                                  if mode == 1:
                                      A meshgrid is generated and cross-section is calculated at each k-point on meshgrid
            mode: control scheme for k_pts sampling
            theta_epsilon, phi_epsilon: same as the definition in function matrix_element_core
            save_file: the saved file name
            
        return:
            (k_pts, matrix_element_list)
            k_pts: a list of k-points
            matrix_element_list: the corresponding list of cross-section
        '''
        return self.matrix_element.matrix_element_in_plane_dependence(hn, k_x_range, k_y_range, mode=mode, theta_epsilon=2*np.pi/9, phi_epsilon=np.pi/2,\
                                          save_file = 'matrix_element.csv')






'''
config = {
    # experimental parameters
    "hn": 80, # in unit of eV
    "inner potential": 15.5, # in unit of eV
    "work function": 4.35, # in unit of eV
    "theta_epsilon": 2*np.pi/9, # in unit of rad
    "phi_epsilon": np.pi/2, # in unit of rad
    # Crystal and DFT parameters
    "c": 4.28e-9, # in the unit of meter
    "k_z list": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # in the unit of \pi/c
    "DFT bands": dft_band_full, # list of DFT bands at various k_z in k_z list
    "orbital component": orbital_component_full, # list of orbital components at various k_z in k_z list
    "atomic number": [63, 63, 48, 48, 48, 48, 33, 33, 33, 33], # list of atomic number
    "orbitals": [["6s", "5py", "5pz", "5px", "4dxy", "4dyz", "4dz2", "4dxz", "4dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["6s", "5py", "5pz", "5px", "4dxy", "4dyz", "4dz2", "4dxz", "4dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["5s", "4py", "4pz", "4px", "4dxy", "4dyz", "4dz2", "4dxz", "4dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["5s", "4py", "4pz", "4px", "4dxy", "4dyz", "4dz2", "4dxz", "4dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["5s", "4py", "4pz", "4px", "4dxy", "4dyz", "4dz2", "4dxz", "4dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["5s", "4py", "4pz", "4px", "4dxy", "4dyz", "4dz2", "4dxz", "4dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["4s", "4py", "4pz", "4px", "3dxy", "3dyz", "3dz2", "3dxz", "3dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["4s", "4py", "4pz", "4px", "3dxy", "3dyz", "3dz2", "3dxz", "3dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["4s", "4py", "4pz", "4px", "3dxy", "3dyz", "3dz2", "3dxz", "3dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["4s", "4py", "4pz", "4px", "3dxy", "3dyz", "3dz2", "3dxz", "3dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"]],
    # matrix element parameters
    "matrix element scenario": 1, # matrix element scenario
    #"Z_eff": 6.95,
    "matrix element calculation indicator": [[0, 0, 0, 0, 0, 0, 0, 0, 0, "max", "max", "max", "max", "max", "max", "max"],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, "max", "max", "max", "max", "max", "max", "max"],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    # simulation parameters
    "simulation e axis": sim_e_axis,
    "simulation k axis": sim_k_axis,
    "self energy real": self_energy_real,
    "self energy img": self_energy_im
}
'''

class spectra_simulation_single_cut:
    def __init__(self, config):
        self.config = config
        self.hn = config['hn']
        self.hn_effective = config['hn'] - config['inner potential'] - config['work function']
        self.theta_epsilon = config['theta_epsilon']
        self.phi_epsilon = config['phi_epsilon']
        
        self.pi_over_c = np.pi/config['c']
        self.kz_list = config['k_z list']
        self.dft_bands_list = config['DFT bands']
        self.orbital_component_list = config['orbital component']

        self.sim_omega_axis = config['simulation e axis']
        self.sim_k_axis = config['simulation k axis']
        if len(set([j.shape for j in self.dft_bands_list]))!=1:
            raise ValueError('--- dft bands in dft_bands_list are not in the same shape ---')            
        if len(self.sim_k_axis) != (self.dft_bands_list[0]).shape[0]:
            raise ValueError('--- Momentum axis for DFT bands and spectra are inconsistent ---')        
        
        self.self_energy_r = config['self energy real']
        self.self_energy_im = config['self energy img']
        
        def eV2J(eV):
            return 1.60218e-19*eV
        def J2eV(J):
            return J/1.60218e-19
        e_mass = 9.1093837e-31; h_bar = 1.054571817e-34
        self.k_0 = np.sqrt(2*e_mass*eV2J(self.hn))/h_bar
        self.v_perp = np.power(h_bar,2)*self.k_0/e_mass
        self.sigma = eV2J(max([self.self_energy_im(j,k) for j in self.sim_k_axis for k in self.sim_omega_axis]))/self.v_perp
        
        print('--- start calculating relevant matrix element ---')
        self.atomic_number = config['atomic number']
        self.orbitals = config['orbitals']
        self.matrix_element_scenario = config['matrix element scenario']
        if self.matrix_element_scenario == 3:
            self.Z_eff = config['Z_eff']
        
        self.matrix_element_calculation_indicator = config['matrix element calculation indicator']
        self.matrix_element_k_axis = np.linspace(self.sim_k_axis[0], self.sim_k_axis[-1], np.int(len(self.sim_k_axis)/50))
        self.matrix_element = np.zeros((len(self.matrix_element_k_axis), len(self.matrix_element_calculation_indicator),\
                                        len(self.matrix_element_calculation_indicator[0])))
        self.matrix_element_config_cache = []; self.matrix_element_cache = []
        for j in range(len(self.matrix_element_calculation_indicator)):
            for k in range(len(self.matrix_element_calculation_indicator[0])):
                if self.matrix_element_calculation_indicator[j][k] == 1:
                    if self.matrix_element_scenario == 1 or 3:
                        config_ = {'Z': self.atomic_number[j], 'orbital': self.orbitals[j][k]}
                        if config in self.matrix_element_config_cache:
                            self.matrix_element[:,j,k] = self.matrix_element_cache[self.matrix_element_config_cache.index(config)]
                        else:
                            matrix_element_calculation = matrix_element(self.matrix_element_scenario, config_)
                            _, matrix_element_val = matrix_element_calculation.matrix_element_in_plane_dependence(self.hn, list(self.matrix_element_k_axis), [0], mode=1,\
                                                                                        theta_epsilon=self.theta_epsilon, phi_epsilon=self.phi_epsilon, save_file = 'matrix_element.csv')
                            matrix_element_val = np.array(matrix_element_val)
                            os.remove("matrix_element.csv")
                            self.matrix_element_config_cache.append(config)
                            self.matrix_element_cache.append(matrix_element_val)
                            self.matrix_element[:,j,k] = matrix_element_val
                    if self.matrix_element_scenario == 2:
                        config = {'Z': self.atomic_number[j], 'orbital': self.orbitals[j][k], "Z_eff": self.Z_eff[j]}
                        if config in self.matrix_element_config_cache:
                            matrix_element_val = self.matrix_element_cache[self.matrix_element_config_cache.index(config)]
                            self.matrix_element[:,j,k] = matrix_element_val
                        else:
                            matrix_element_calculation = matrix_element(self.matrix_element_scenario, config)
                            _, matrix_element_val = matrix_element_in_plane_dependence(self.hn, self.matrix_element_k_axis, [0], mode=1, \
                                                                                        theta_epsilon=self.theta_epsilon, phi_epsilon=self.phi_epsilon, save_file = 'matrix_element.csv')
                            matrix_element_val = np.array(matrix_element_val)
                            os.remove("matrix_element.csv")
                            self.matrix_element_config_cache.append(config)
                            self.matrix_element_cache.append(matrix_element_val)
                            self.matrix_element[:,j,k] = matrix_element_val
        for j in range(len(self.matrix_element_calculation_indicator)):
            for k in range(len(self.matrix_element_calculation_indicator[0])):
                if self.matrix_element_calculation_indicator[j][k] == "max":
                    for l in range(self.matrix_element.shape[0]):
                        self.matrix_element[l,j,k] = np.amax(self.matrix_element[l,:,:])
        
        matrix_element_full = np.zeros((len(self.sim_k_axis), len(self.matrix_element_calculation_indicator), len(self.matrix_element_calculation_indicator[0])))
        for j in range(matrix_element_full.shape[0]):
            for k in range(matrix_element_full.shape[1]):
                for l in range(matrix_element_full.shape[2]):
                    matrix_element_full[j,k,l] = np.interp(self.sim_k_axis[j], self.matrix_element_k_axis, self.matrix_element[:,k,l])
        self.matrix_element = copy.deepcopy(matrix_element_full)
        del matrix_element_full, self.matrix_element_config_cache, self.matrix_element_cache
        print("--- matrix element calculation finished! ---")
        
    def A_i(self, kz_idx):
        '''
        Simulate spectra at specific k_z.
        params:
            kz_idx: the index in k_z list
        return:
            spectra: np.array(len(self.sim_k_axis), len(self.sim_omega_axis))
        '''
        band = self.dft_bands_list[kz_idx]
        spectra = np.zeros((len(self.sim_k_axis), len(self.sim_omega_axis)))
        for j in range(spectra.shape[0]): # momentum
            momentum = self.sim_k_axis[j]
            for k in range(spectra.shape[1]): # eigenenergies
                epsilon_k = band[j,k]
                matrix_element_weight = np.sum(np.multiply(self.matrix_element[j,:,:], self.orbital_component_list[kz_idx][j,k,:,:]))
                for l in range(spectra.shape[1]): #omega
                    omega = self.sim_omega_axis[l]
                    spectra[j,l] += matrix_element_weight*self.self_energy_im(momentum, omega)\
                    /(np.power(omega-epsilon_k-self.self_energy_r(momentum,omega),2) + np.power(self.self_energy_im(momentum, omega),2))
        spectra /= np.sum(spectra)
        return spectra
    
    def A_kz_int(self, save_file = "spectra.csv"):
        if len(self.kz_list)==0:
            sim_spectra_norm = self.A_i(0)
            np.savetxt(save_file, sim_spectra_norm, delimiter=',')
            return sim_spectra_norm
        
        self.k_perp_idx_seed = list(np.arange(0, len(self.kz_list), 1))+list(np.arange(len(self.dft_bands_list)-2, 0, -1))
        self.sim_spectra = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.A_i)(j) for j in self.k_perp_idx_seed)
        k_perp_list = np.arange(1,100*len(self.kz_list),1)
        self.kz_weight = [np.power(self.sigma,2)/(np.power(k_perp*self.pi_over_c/(len(self.kz_list)-1)-self.k_0,2)+np.power(self.sigma,2)) for k_perp in k_perp_list]
        self.kz_weight_folded = np.zeros(len(self.k_perp_idx_seed))
        
        sim_spectra_kz_int = np.zeros((len(self.sim_k_axis), len(self.sim_omega_axis)))
        for j in range(len(k_perp_list)):
            sim_spectra_kz_int += self.kz_weight[j]*self.sim_spectra[j%len(self.k_perp_idx_seed)]
            self.kz_weight_folded[j%len(self.k_perp_idx_seed)] += self.kz_weight[j]
        
        sim_spectra_norm = sim_spectra_kz_int/np.sum(sim_spectra_kz_int)
        np.savetxt(save_file, sim_spectra_norm, delimiter=',')
        return sim_spectra_norm
    
    def analysis_plot_kz_weight(self):
        plt.plot(np.array(self.k_perp_idx_seed)/len(self.kz_list), self.kz_weight_folded/np.sum(self.kz_weight_folded))
        plt.title("contributions at different $k_z$")
        plt.xlabel("$k_z$ (in the unit of $\pi/(nc)$)")
        plt.ylabel("weight")
        
    def update_self_energy(self, self_energy_r, self_energy_im):
        self.self_energy_r = self_energy_r
        self.self_energy_im = self_energy_im




f = h5py.File('case_EuCd2As2.h5', 'r')
dft_band_full = [np.array(f["dft_bands"][str(j)]) for j in range(11)]
orbital_component_full = [np.array(f["orbital_components"][str(j)]) for j in range(11)]
f.close()

plt.figure(figsize=(24,6))
for idx in range(11):
    plt.subplot(1,11,idx+1)
    for band_idx in range(dft_band_full[idx].shape[1]):
        plt.plot(dft_band_full[idx][:, band_idx], c='r')
    
    plt.ylim(-3,0)
    plt.title(''.join(['$k_z=$', str(idx), '$/10(\pi/c)$']))


sim_e_axis = list(np.linspace(-1, 0, 50))
sim_k_axis = list(np.linspace(-0.9434, 0.9434, dft_band_full[0].shape[0])) # for EuCd2As2

w_ar=np.linspace(-3,.1,1000)
def self_energy_im_seed(w):
    g = 0.001
    return 2*g*(1/(1+np.exp(100*w))-.5)
self_energy_im_ar = self_energy_im_seed(w_ar)
self_energy_r_ar = ft.hilbert(self_energy_im_seed(w_ar))

def self_energy_im(k, omega):
    return np.interp(omega, w_ar, self_energy_im_ar)

def self_energy_real(k, omega):
    return np.interp(omega, w_ar, self_energy_r_ar)

config = {
    # experimental parameters
    "hn": 80,
    "inner potential": 15.5, # 15.5 for ECA, 13.5 for ECP
    "work function": 4.35,
    "theta_epsilon": 2*np.pi/9,
    "phi_epsilon": np.pi/2,
    # Crystal and DFT parameters
    "c": 4.28e-9, # in the unit of meter. ECA: 4.28e-9; ECP: 3.589e-9
    "k_z list": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # in the unit of \pi/c
    "DFT bands": dft_band_full,
    "orbital component": orbital_component_full,
    "atomic number": [63, 63, 48, 48, 48, 48, 33, 33, 33, 33],
    "orbitals": [["6s", "5py", "5pz", "5px", "4dxy", "4dyz", "4dz2", "4dxz", "4dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["6s", "5py", "5pz", "5px", "4dxy", "4dyz", "4dz2", "4dxz", "4dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["5s", "4py", "4pz", "4px", "4dxy", "4dyz", "4dz2", "4dxz", "4dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["5s", "4py", "4pz", "4px", "4dxy", "4dyz", "4dz2", "4dxz", "4dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["5s", "4py", "4pz", "4px", "4dxy", "4dyz", "4dz2", "4dxz", "4dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["5s", "4py", "4pz", "4px", "4dxy", "4dyz", "4dz2", "4dxz", "4dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["4s", "4py", "4pz", "4px", "3dxy", "3dyz", "3dz2", "3dxz", "3dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["4s", "4py", "4pz", "4px", "3dxy", "3dyz", "3dz2", "3dxz", "3dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["4s", "4py", "4pz", "4px", "3dxy", "3dyz", "3dz2", "3dxz", "3dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"],
                 ["4s", "4py", "4pz", "4px", "3dxy", "3dyz", "3dz2", "3dxz", "3dx2-y2", "4fy3x2", "4fxyz", "4fyz2", "4fz3", "4fxz2", "4fzx2", "4fx3"]],
    # matrix element parameters
    "matrix element scenario": 1,
    #"Z_eff": 6.95,
    "matrix element calculation indicator": [[0, 0, 0, 0, 0, 0, 0, 0, 0, "max", "max", "max", "max", "max", "max", "max"],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, "max", "max", "max", "max", "max", "max", "max"],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    # simulation parameters
    "simulation e axis": sim_e_axis,
    "simulation k axis": sim_k_axis,
    "self energy real": self_energy_real,
    "self energy img": self_energy_im
}

spectra_simulation = spectra_simulation_single_cut(config)
spectra = spectra_simulation.A_kz_int()
