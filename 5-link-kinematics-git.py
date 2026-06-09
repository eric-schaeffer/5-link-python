#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:36:01 2026

@author: eric
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider
import time

# ==========================================
# BLOCK 0: EINGABEN FAHRWERKGROESSEN FUER SIMULATION
# ==========================================

z_min = - 100
z_max = 100
z_step = 10

zDot = 1000
z2Dot = 0

r = 250

F_w = [0, -10000, 0]
F_t = [0, 0, 0]



# fuer Std Axis
steer_min_STD = - 25
steer_max_STD = 25
steer_step_STD = 2.5

# fuer Std Axis
steerDot_STD = 100
steer2Dot_STD = 0

# fuer Std Axis
P_a0P0_STD = np.array([0,25,-20])
P_a1P0_STD = np.array([-50,0,-150])
P_a2P0_STD = np.array([300,0,-150])
P_a3P0_STD = np.array([275,50,120])
P_a4P0_STD = np.array([-75,50,120])
P_a5_STD = np.array([125,0,350])
P_aiP0_STD = [P_a0P0_STD, P_a1P0_STD, P_a2P0_STD, P_a3P0_STD, P_a4P0_STD, P_a5_STD]

# fuer Std Axis
r_bP0_STD = np.array([105,400,0]) # z-Wert MUSS 0 sein!!!
P_b0P0_STD = np.array([25,320,0])
P_b1P0_STD = np.array([110,340,-150])
P_b2P0_STD = np.array([140,340,-150])
P_b3P0_STD = np.array([115,300,150])
P_b4P0_STD = np.array([85,300,150])
P_b5P0_STD = np.array([0,-150,0])+r_bP0_STD
P_b6P0_STD = np.array([-0.5,-100,-2])+r_bP0_STD # mit x- und z-Werten im np.array aendert man statischen Toe und statischen Camber
P_biP0_STD = [P_b0P0_STD, P_b1P0_STD, P_b2P0_STD, P_b3P0_STD, P_b4P0_STD, P_b5P0_STD, P_b6P0_STD]

# fuer Std Axis
a0_STD = 1
a1_STD = 0
a2_STD = 0
a3_STD = 0
a4_STD = 0
ai_STD = [a0_STD, a1_STD, a2_STD, a3_STD, a4_STD]



# fuer Rot Axis
steer_min_ROT = - 0.75
steer_max_ROT = 0.20
steer_step_ROT = 0.05

# fuer Rot Axis
steerDot_ROT = 3
steer2Dot_ROT = 3

# fuer Rot Axis
P_a0P0_ROT = np.array([-350,500,15]) # P1
P_a1P0_ROT = np.array([175,300,-92]) # P3
P_a2P0_ROT = np.array([-50,330,-79]) # P5
P_a3P0_ROT = np.array([-50,330,295]) # P7
P_a4P0_ROT = np.array([175,300,280]) # P9
P_a5_ROT = np.array([125,350,520]) # P11
P_aiP0_ROT = [P_a0P0_ROT, P_a1P0_ROT, P_a2P0_ROT, P_a3P0_ROT, P_a4P0_ROT, P_a5_ROT]

# fuer Rot Axis
r_aiP0_ROT = np.array([35, 350, 70])
r_aiP0_ROT = [r_aiP0_ROT, r_aiP0_ROT, r_aiP0_ROT, r_aiP0_ROT, r_aiP0_ROT]

# fuer Rot Axis
r_bP0_ROT = np.array([125,750,0]) # Pwc, z-Wert MUSS 0 sein!!!
P_b0P0_ROT = np.array([125,670,0]) # P2
P_b1P0_ROT = np.array([175,665,-90]) # P4
P_b2P0_ROT = np.array([75,640,-80]) # P6
P_b3P0_ROT = np.array([75,615,310]) # P8
P_b4P0_ROT = np.array([175,640,305]) # P10
P_b5P0_ROT = np.array([125,600,100]) # P12
P_b6P0_ROT = np.array([0,-50,0])+r_bP0_ROT # mit x- und z-Werten im np.array aendert man statischen Toe und statischen Camber
P_biP0_ROT = [P_b0P0_ROT, P_b1P0_ROT, P_b2P0_ROT, P_b3P0_ROT, P_b4P0_ROT, P_b5P0_ROT, P_b6P0_ROT]

# fuer Rot Axis
u_iUnscaled = np.array([-0.20, -0.08, 1])
u_iScaled = u_iUnscaled / np.linalg.norm(u_iUnscaled)
u_i = [u_iScaled, u_iScaled, u_iScaled, u_iScaled, u_iScaled]
       
# fuer Rot Axis
a0_ROT = 0
a1_ROT = 1
a2_ROT = 1
a3_ROT = 1
a4_ROT = 1
ai_ROT = [a0_ROT, a1_ROT, a2_ROT, a3_ROT, a4_ROT]



config = 0 # 0 fuer Std Axis, 1 fuer Rot Axis



class kinematics:

    def __init__(self):
        self.ai_c_ai = None
        self.b_c_bi = None
        self.d_i = None
        self.c_ai = None
        self.P_ai = None
        self.c_bi = None
        self.l_i = None
        self.beta_val = None
        self.r_b = None
        self.Gs_a = None
        self.Gs_b_A = None
        self.Gs_b_k = None
        self.eta_val = None
        self.P_bi = None
        self.r_bw = None
        self.omega_b = None
        self.v_b = None
        self.eta_val = None
        self.etaDot_val = None
        self.camber = None
        self.toe = None
        self.trail = None
        self.scrub = None
        self.caster_trail = None
        self.kingpin_offset_hub = None
        self.caster = None
        self.kpi = None
        self.laengspol = None
        self.querpol = None
        self.Fs = None
        self.laengsTransl = None
        self.querTransl = None




    # ==========================================
    # BLOCK 1: METHODEN FUER LAGEEBENE
    # ==========================================

    def _0ai_T(self, phi_i, u_i):
        # Rotaionsmatrix mit Euler-Parametern
        if phi_i == 0:
            R_i = np.eye(3)
        else:
            p_i_tilde = np.cross(np.eye(3), u_i * np.sin(phi_i*0.5))
            R_i = np.eye(3) + 2 * np.cos(phi_i*0.5) * p_i_tilde + 2 * p_i_tilde @ p_i_tilde
        return R_i


    def _0b_T(self, phi, theta, psi):
        # Rotationsmatrix mit Kardan-Winkeln
        c_ph = np.cos(phi)
        c_th = np.cos(theta)
        c_ps = np.cos(psi)
        s_ph = np.sin(phi)
        s_th = np.sin(theta)
        s_ps = np.sin(psi)
        return np.array([
            [c_ps*c_th, -s_ps*c_ph+c_ps*s_th*s_ph, s_ps*s_ph+c_ps*s_th*c_ph],
            [s_ps*c_th, c_ps*c_ph+s_ps*s_th*s_ph, -c_ps*s_ph+s_ps*s_th*c_ph],
            [-s_th, c_th*s_ph, c_th*c_ph]
            ])


    # Aufstellen der Schliessbedingungen auf Lageebene in Matrizenform
    def c_bi_func(self, phi, theta, psi):
        return (self._0b_T(phi, theta, psi) @ self.b_c_bi.T).T
    def l_i_func(self, phi, theta, psi, x, y, z):
        c_bi_func = self.c_bi_func(phi, theta, psi)
        return np.array([x, y, z]) + c_bi_func[:5] - self.P_ai[:5] # nur die 5 Lenker Anbindungen
    def gs_i(self, beta, z):
        l_i_funcZ = self.l_i_func(*beta, z)
        return np.sum(l_i_funcZ**2, axis=1) - self.d_i**2 # entspricht Skalarprodukt, axis=1 heisst entlang der Spalten



    # ==========================================
    # BLOCK 2: SETUP GEOMETRIE FUER LAGEEBENE (Wird bei Lenk-Update aufgerufen)
    # ==========================================

    def set_geometry(self, P_aiP0, P_biP0, r_aiP0, r_aiDelta, r_bP0, phi_i, u_i):
        # Matrizen c_ai und c_bi in eigenen koerperfesten Koordinatensystemen (KS) in Konstruktionslage (z=0)
        self.ai_c_ai = np.array(P_aiP0)[:5] - np.array(r_aiP0)
        self.b_c_bi = np.array(P_biP0) - np.array(r_bP0)
        
        # Lenkerlaenge in Konstruktionslage (z=0)
        self.d_i = np.linalg.norm(np.array(P_biP0)[:5] - np.array(P_aiP0)[:5], axis=1) # entspricht Betrag der einzelnen Vektoren
        
        # Rotation c_ai in raumfestes KS
        self.c_ai = np.array([
            self._0ai_T(np.array(phi_i)[i], np.array(u_i)[i]) @ self.ai_c_ai[i]
            for i in range(0, 5)
            ])
        
        # Berechnung Anbindungspunkte am Chassis inkl. Anbindungspunkt Feder-Daempfer-Einheit
        P_ai = np.array(r_aiP0) + np.array(r_aiDelta) + self.c_ai
        self.P_ai = np.vstack([P_ai, np.array(P_aiP0)[5]])



    # ==========================================
    # BLOCK 3: LAGEEBENE LOESEN (z-Bewegung)
    # ==========================================

    def beta_calc(self, beta_start, z):
        # Loesen der Schliessbedingungen auf Lageebene
        beta_val, _, ier, mesg = fsolve(self.gs_i, np.array(beta_start), args=(z,), full_output=True)
        
        if ier == 1: # Ueberpruefung der Art der Loesung bzw. ob eine Loesung gefunden wurde
            self.beta_val = beta_val.copy()
            self.r_b = np.concatenate((beta_val[3:5], [z]))
        else:
            print(f"Warnung: fsolve hat keine Loesung gefunden! Grund: {mesg}")


    def calc_position(self, z, r):
        self.l_i = self.l_i_func(*self.beta_val, z)
        self.c_bi = self.c_bi_func(*self.beta_val[:3])
        
        # Berechnung Anbindungspunkte am Radtraeger inkl. Anbindungspunkt Feder-Daempfer-Einheit und Verbindungspunkt Radtraeger
        self.P_bi = self.r_b + self.c_bi
        
        # Berechnung Vektor von Radmittelpunkt zu Radaufstandspunkt
        c_b6 = self.c_bi[6]
        z6 = c_b6[2]
        if z6 == 0:
            self.r_bw = r * np.array([0,0,-1])
        else:
            zVal = - (c_b6[0]**2 + c_b6[1]**2) / c_b6[2]
            r_bw_part = np.concatenate((c_b6[:2], [zVal]))
            self.r_bw = r / np.linalg.norm(r_bw_part) * r_bw_part
        if z6 < 0:
            self.r_bw = - self.r_bw



    # ==========================================
    # BLOCK 4: GESCHWINDIGKEITS- & BESCHLEUNIGUNGSEBENE LOESEN
    # ==========================================

    def Gs_a_calc(self):
        # Berechnung Bindungsmatrix Gs_a
        Gs_a_part = np.cross(self.l_i, self.c_ai)
        self.Gs_a = np.hstack([Gs_a_part, -self.l_i])


    def Gs_b_calc(self):
        # Berechnung Bindungsmatrix Gs_b
        Gs_b_part = -np.cross(self.l_i, self.c_bi[:5])
        Gs_b = np.hstack([Gs_b_part, self.l_i])
        # Aufteilung Bindungsmatrix fuer spaetere Berechnungen
        self.Gs_b_A = Gs_b[:, :5]
        self.Gs_b_k = Gs_b[:, 5]


    def eta_calc(self, zDot, i, omega_ai, v_ai):
        eta_ai = np.hstack([np.array(omega_ai), np.array(v_ai)])
        Gs_ai_mult_eta_ai = np.sum(self.Gs_a * eta_ai, axis=1) # entspricht Skalarprodukt
        
        if i == 0: # allgemeine Momentane Schraubachse
            b = -self.Gs_b_k * zDot - Gs_ai_mult_eta_ai
        elif i == 1: # nur Federung Momentane Schraubachse
            b = -self.Gs_b_k * zDot
        elif i == 2: # nur Lenkung Momentane Schraubachse
            b = - Gs_ai_mult_eta_ai
        
        # Loesen des LGS nach eta
        self.eta_val = np.linalg.solve(self.Gs_b_A, b)


    def etaDot_calc(self, zDot, z2Dot, omega_ai, v_ai, omegaDot_ai, vDot_ai):
        etaDot_ai = np.hstack([np.array(omegaDot_ai), np.array(vDot_ai)])
        Gs_ai_mult_etaDot_ai = np.sum(self.Gs_a * etaDot_ai, axis=1) # entspricht Skalarprodukt
        
        self.omega_b = self.eta_val[:3]
        self.v_b = np.concatenate((self.eta_val[3:5], [zDot]))
        
        # Berechnung gammas_ab
        cDot_b = np.cross(self.omega_b, self.c_bi[:5])
        cDot_ai = np.cross(np.array(omega_ai), self.c_ai)
        omega_b_cross_cDot_b = np.cross(self.omega_b, cDot_b)
        omega_ai_cross_cDot_ai = np.cross(np.array(omega_ai), cDot_ai)
        lDot_i = self.v_b + cDot_b - np.array(v_ai) - cDot_ai
        part0 = np.sum(self.l_i * (omega_b_cross_cDot_b - omega_ai_cross_cDot_ai), axis=1) # entspricht Skalarprodukt
        part1 = np.sum(lDot_i * lDot_i, axis=1) # entspricht Skalarprodukt
        gammas_ab = part0 + part1
        
        b = -self.Gs_b_k * z2Dot - gammas_ab - Gs_ai_mult_etaDot_ai
        
        # Loesen des LGS nach etaDot
        self.etaDot_val = np.linalg.solve(self.Gs_b_A, b)



    # ==========================================
    # BLOCK 5: FAHRWERK KENNGROESSEN
    # ==========================================

    def kinematic_properties_calc(self, z, omega_ai, v_ai):
        # Berechnung Camber und Toe
        c_b6 = self.c_bi[6]
        self.camber = np.pi * 0.5 - np.arccos(1 / np.linalg.norm(c_b6) * c_b6 @ np.array([0,0,1]))
        c_b6Z0 = np.concatenate((c_b6[:2], [0]))
        self.toe = np.arccos(1 / np.linalg.norm(c_b6Z0) * c_b6Z0 @ np.array([1,0,0])) - np.pi * 0.5
        
        # reine Lenkbewegung
        self.eta_calc(0, 2, omega_ai, v_ai) # momentane Schraubachse, 2 = reine Lenkbewegung
        omega_b_steer = self.eta_val[:3]
        v_b_steer = np.concatenate((self.eta_val[3:5], [0]))
        
        # Punkt auf der momentanen Schraubachse
        r_P = self.r_b + (np.cross(omega_b_steer, v_b_steer) / np.linalg.norm(omega_b_steer)**2)
        
        # b_vertikalpol und w_vertikalpol berechnen
        if np.all(omega_b_steer == np.zeros(3)):
            print("Warnung: reine Translation Vertikalpol")
        else:
            t = (self.r_b[2] - r_P[2]) / omega_b_steer[2]
            r_b_vertikalpol = r_P + t * omega_b_steer
            b_vertikalpol =  self.r_b - r_b_vertikalpol
            t = (self.r_b[2] + self.r_bw[2] - r_P[2]) / omega_b_steer[2]
            r_w_vertikalpol = r_P + t * omega_b_steer
            w_vertikalpol =  self.r_b + self.r_bw - r_w_vertikalpol
            
            # fuer Darstellung approximierte momentane Lenkachse
            self.rot_axis_low = r_w_vertikalpol
            self.rot_axis_high = (r_b_vertikalpol - r_w_vertikalpol) * 2 + r_w_vertikalpol
        
        # Kenngroessen zuweisen
        self.trail = w_vertikalpol[0] * (-1)
        self.scrub = w_vertikalpol[1]
        self.caster_trail = b_vertikalpol[0] * (-1)
        self.kingpin_offset_hub = b_vertikalpol[1]
        
        # Rotation des Vektors mit Rad fuer Kenngroessen Berechnung, mit b_vertikalpol_rot und w_vertikalpol_rot, da sich ansonsten bei Richtungsaenderung Lenkgeschwindigkeit auch die Caster und KPI Wert *(-1)
        img_steer_axis = r_b_vertikalpol - r_w_vertikalpol
        
        omega_steer_y0 = np.array([img_steer_axis[0], 0, img_steer_axis[2]])
        self.caster = (np.pi * 0.5 - np.arccos(1 / np.linalg.norm(omega_steer_y0) * omega_steer_y0 @ np.array([1,0,0]))) * (-1)
        omega_steer_x0 = np.array([0, img_steer_axis[1], img_steer_axis[2]])
        self.kpi = (np.pi * 0.5 - np.arccos(1 / np.linalg.norm(omega_steer_x0) * omega_steer_x0 @ np.array([0,1,0]))) * (-1)


    def instant_centers(self, z):
        
        # reine Federbewegung
        self.eta_calc(100, 1, np.zeros((5, 3)), np.zeros((5, 3))) # momentane Schraubachse, 1 = reine Federbewegung, 100 genauer Wert ist egal
        omega_b_z = self.eta_val[:3]
        v_b_z = np.concatenate((self.eta_val[3:5], [100]))
        
        # Laengspol berechnen
        v_b_z_y0 = np.array([v_b_z[0], 0, v_b_z[2]])
        omega_b_z_x0_z0 = np.array([0, omega_b_z[1], 0])
        if omega_b_z[1] == 0:
            self.laengsTransl = 1
            # print("Warnung: reine Translation Laengspol", z, omega_b_z)
        else:
            self.laengsTransl = 0
            self.laengspol = (np.cross(omega_b_z_x0_z0, v_b_z_y0) / np.linalg.norm(omega_b_z_x0_z0)**2) + self.r_b
        
        # Querpol berechnen
        v_b_zx0 = np.array([0, v_b_z[1], v_b_z[2]])
        omega_b_z_y0_z0 = np.array([omega_b_z[0], 0, 0])
        if omega_b_z[0] == 0:
            self.querTransl = 1
            # print("Warnung: reine Translation Querpol", z, omega_b_z)
        else:
            self.querTransl = 0
            self.querpol = (np.cross(omega_b_z_y0_z0, v_b_zx0) / np.linalg.norm(omega_b_z_y0_z0)**2) + self.r_b
        
        # self.v_w = v_b_z + np.cross(omega_b_z, self.r_bw)



    # ==========================================
    # BLOCK 6: FÜR DIE KENNGROESSEN-PLOTS EINE METHODE
    # ==========================================

    def calc_geometry(self, z, beta_start, omega_ai, v_ai, r):
        # Berechnungen fuer die Darstellung als 3d Plot und der Kenngroessen als 2d Plot
        self.beta_calc(beta_start, z)
        self.calc_position(z, r)
        self.Gs_a_calc()
        self.Gs_b_calc()
        self.kinematic_properties_calc(z, omega_ai, v_ai)



    # ==========================================
    # BLOCK 7: KRAEFTE-BERECHNUNG
    # ==========================================

    def link_force_calc(self, F_w, F_t):
        # Matrix der Vektoren der Lenker fuer die 5 Stablenker inkl. Feder-Daempfer-Einheit
        l_i = np.vstack((self.l_i, self.r_b + self.c_bi[5, :] - self.P_ai[5, :]))
        
        norms = np.linalg.norm(l_i, axis=1, keepdims=True) # entspricht Betrag der einzelnen Vektoren
        E = - l_i / norms # Matrix der Einheitsvektoren 6, 3)

        X_i = np.cross(self.c_bi[:6,:], E) # Matrix der Einheitsmomentenvektoren (6, 3)
        E_i_X_i_matrix = np.vstack((E.T, X_i.T)) # Matrix der Einheitsvektoren und Einheitsmomentenvektoren (6, 6)
        
        F_w_vec = np.array(F_w)
        F_t_vec = np.array(F_t)
        
        # if self.c_bi[6,1] == 0:
        #     print('Hinweis Kraftberechnung Tangentialkraft F_t ApproX_imation, Grund: Lenkwinkel genau 90 Grad')
        #     direction = np.array([1, -self.c_bi[6,0]/-0.000001, 0])
        #     F_t_vec = F_t * direction / np.linalg.norm(direction)
        # elif self.c_bi[6,1] >= 0:
        #     direction = np.array([-1, self.c_bi[6,0]/self.c_bi[6,1], 0])
        #     F_t_vec = F_t * direction / np.linalg.norm(direction)
        # else:
        #     direction = np.array([1, -self.c_bi[6,0]/self.c_bi[6,1], 0])
        #     F_t_vec = F_t * direction / np.linalg.norm(direction)
        
        M_w = np.cross(self.r_bw, F_w_vec)
        F_w_b = np.concatenate((-F_w_vec -F_t_vec, -M_w))
        
        # Loesen des LGS nach den Lenkerkraeften
        self.Fs = np.linalg.solve(E_i_X_i_matrix, F_w_b)
        
        # self.check = E @ self.Fs





class liveMovementCommon:

    def __init__(self, kine_object):
        self.kine = kine_object
        self.z_array = None
        self.steer_array = None
        self.beta_start = None
        self.beta_matrix = None
        self.beta_list = None
        
        # Dictionary fuer alle Kenngroessen
        self.results = {
            'beta': None, 'caster': None, 'kpi': None, 'toe': None, 
            'camber': None, 'trail': None, 'scrub': None, 
            'caster_trail': None, 'kingpin_offset_hub': None
        }
        
        # fuer Zustand bei start
        self.steer_index = 0
        self.z_index = 0
        self.z0_index = 0
        
        # Plot-Objekte
        self.fig = None
        self.ax_3d = None
        self.axes_2d = []
        self.plot_configs = [
            ("toe", "toe"), ("camber", "camber"), 
            ("caster", "caster"), ("kpi", "kpi"),
            ("trail", "trail"), ("scrub", "scrub"),
            ("caster_trail", "caster_trail"), ("kingpin_offset_hub", "kingpin_offset_hub")
        ]



    # ==========================================
    # BLOCK 8: HILFSFUNKTION WERTE AUSLESEN
    # ==========================================

    def extract_kinematic_vals(self, z):
        # Kenngroessen aus self.kine.
        return {
            'beta': np.concatenate((self.kine.beta_val, [z])),
            'caster': np.degrees(self.kine.caster),
            'kpi': np.degrees(self.kine.kpi),
            'toe': np.degrees(self.kine.toe),
            'camber': np.degrees(self.kine.camber),
            'trail': self.kine.trail,
            'scrub': self.kine.scrub,
            'caster_trail': self.kine.caster_trail,
            'kingpin_offset_hub': self.kine.kingpin_offset_hub
        }



    # ==========================================
    # BLOCK 9: GRUNDGERUEST PLOTS
    # ==========================================

    def setup_plot(self, P_aiP0, P_biP0, r_bP0, r):
        start_time = time.time()
        
        plt.ion() # Interaktiver Modus fuer Live Updates
        
        # Layout der figure [ 3D-Plot (breit) | 4 x 2D-Plots (schmal) | 4 x 2D-Plots (schmal) ]
        self.fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(4, 3, width_ratios=[2, 1, 1])
        self.ax_3d = self.fig.add_subplot(gs[:, 0], projection='3d')
        self.axes_2d = []
        
        # Initialisierung der 8 Subplots in den rechten beiden Spalten
        self.plot_configs = [
            ("toe", "toe"), ("camber", "camber"), 
            ("caster", "caster"), ("kpi", "kpi"),
            ("trail", "trail"), ("scrub", "scrub"),
            ("caster_trail", "caster_trail"), ("kingpin_offset_hub", "kingpin_offset_hub")
        ]
        for i, (label, attr) in enumerate(self.plot_configs):
            row = i % 4
            col = 1 if i < 4 else 2
            ax = self.fig.add_subplot(gs[row, col])
            ax.set_xlabel("z [mm]")
            ax.grid(True)
            self.axes_2d.append(ax)
        
        plt.tight_layout()
        
        # Berechnungen fuer Limits der Achsen
        self.xMax = max(np.array(P_aiP0)[:, 0].max(), np.array(P_biP0)[:, 0].max()) + 100
        self.xMin = min(np.array(P_aiP0)[:, 0].min(), np.array(P_biP0)[:, 0].min()) - 100
        self.yMax = r_bP0[1] + 100
        self.yMin = np.array(P_aiP0)[:, 1].min() - 100
        self.z_max = max(max(np.array(P_aiP0)[:5, 2].max(), np.array(P_biP0)[:, 2].max(), r_bP0[2]) + 200, np.array(P_aiP0)[5, 2] + 50)
        self.z_min = min(np.array(P_aiP0)[:, 2].min(), np.array(P_biP0)[:, 2].min(), r_bP0[2] - r) - 200
        
        end_time = time.time()
        dauer = (end_time - start_time) * 1000
        
        # fuer die Darstellung in der Konsole
        print(f"{'Dauer Plot Setup':<31} | {dauer:.3f} ms")
        print("="*90 + "\n")



    # ==========================================
    # BLOCK 10: AKTUALISIERUNG PLOTS
    # ==========================================

    def update_plot(self):
        start_time = time.time()
        
        # 3d-Achsen initialisieren
        self.ax_3d.cla()
        self.ax_3d.set_xlim([self.xMin, self.xMax])
        self.ax_3d.set_ylim([self.yMin, self.yMax])
        self.ax_3d.set_zlim([self.z_min, self.z_max])
        self.ax_3d.set_xlabel('X [mm]')
        self.ax_3d.set_ylabel('Y [mm]')
        self.ax_3d.set_zlabel('Z [mm]')
        
        # Gelenke der 5 Stablenker als Punkte plotten
        for point in self.kine.P_ai:
            self.ax_3d.scatter(*point, color='black', s=50)
        for point in self.kine.P_bi[:6]:
            self.ax_3d.scatter(*point, color='black', s=50)
        
        # die 5 Stablenker als Linien plotten
        for i in range(5):
            P_a = self.kine.P_ai[i]
            P_b = self.kine.P_bi[i]
            self.ax_3d.plot(
                [P_a[0], P_b[0]],
                [P_a[1], P_b[1]],
                [P_a[2], P_b[2]],
                color='black'
            )
            
        # Feder-Daempfer-Element als lila Linie plotten
        self.ax_3d.plot(
            [self.kine.P_bi[5, 0], self.kine.P_ai[5, 0]],
            [self.kine.P_bi[5, 1], self.kine.P_ai[5, 1]],
            [self.kine.P_bi[5, 2], self.kine.P_ai[5, 2]],
            color='purple'
        )
        
        # Radtraeger plotten als Wire Frame mit P_b6 als Verbindungspunkt zu den anderen P_bi
        P_b6 = self.kine.P_bi[6]
        for i in range(6):
            self.ax_3d.plot(
                [self.kine.P_bi[i, 0], P_b6[0]],
                [self.kine.P_bi[i, 1], P_b6[1]],
                [self.kine.P_bi[i, 2], P_b6[2]],
                color='blue',
            )
        
        # Drehachse des Rads am Radtraeger als gruene Linie plotten
        self.ax_3d.plot(
            [self.kine.r_b[0], P_b6[0]], 
            [self.kine.r_b[1], P_b6[1]], 
            [self.kine.r_b[2], P_b6[2]],
            color='green'
        )
        
        # Verbindungslinie Radmittelpunkt zu Radaufstandspunkt als rote Linie plotten
        self.ax_3d.plot(
            [self.kine.r_b[0], self.kine.r_b[0] + self.kine.r_bw[0]], 
            [self.kine.r_b[1], self.kine.r_b[1] + self.kine.r_bw[1]], 
            [self.kine.r_b[2], self.kine.r_b[2] + self.kine.r_bw[2]], 
            color='red'
        )
        
        # Radmittelpunkt als gruenen Punkt und Radaufstandspunkt als roten Punkt plotten
        self.ax_3d.scatter(*self.kine.r_b, color='green', s=50)
        self.ax_3d.scatter(*(self.kine.r_b + self.kine.r_bw), color='red', s=50)
        
        # approximierte Lenkachse Plotten
        self.ax_3d.plot(
            [self.kine.rot_axis_low[0], self.kine.rot_axis_high[0]],
            [self.kine.rot_axis_low[1], self.kine.rot_axis_high[1]],
            [self.kine.rot_axis_low[2], self.kine.rot_axis_high[2]],
            color='cyan'
        )
        
        # die Kenngroessen Kurven plotten
        for i, ax in enumerate(self.axes_2d):
            ax.cla()
            label, dict_key = self.plot_configs[i]
            x_data = self.results[dict_key]
            unit = "°" if i < 4 else "mm"
            ax.plot(self.z_array, x_data, color='blue', label=label)
            ax.set_ylabel(f"{label} [{unit}]")
            ax.set_xlabel("z [mm]")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True)
        
        # Figure updaten
        self.ax_3d.set_aspect('equal')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
        end_time = time.time()
        dauer = (end_time - start_time) * 1000
        
        # fuer die Darstellung in der Konsole
        print(f"{'Dauer Plot Update':<17} | {dauer:>8.2f} ms")
        print("="*90 + "\n")



    # ==========================================
    # BLOCK 11: VERWANDELN IN INTERAKTIVEN PLOT
    # ==========================================

    def add_sliders(self, z_min, z_max, z_step, steer_min, steer_max, steer_step, F_w, F_t, *args):
        # positionieren und erstellen der Slider
        self.fig.subplots_adjust(bottom=0.15)
        ax_z = self.fig.add_axes([0.2, 0.08, 0.6, 0.02])
        ax_steer = self.fig.add_axes([0.2, 0.03, 0.6, 0.02])
        self.z_slider = Slider(ax_z, 'z_curr [mm]', z_min, z_max, valinit=0.0, valfmt='%1.2f')
        self.steer_slider = Slider(ax_steer, 'steer_curr [rad or mm]', steer_min, steer_max, valinit=0.0, valfmt='%1.2f')

        # speichern aktuelle Werte
        self.z_curr = self.z_slider.val
        self.steer_curr = self.steer_slider.val

        # beim Loslassen der Maustaste beim Verschieben eines Sliders wird die Berechnung und der Plot aktualisiert
        def update_on_release(event=None):
            z_val = self.z_slider.val
            steer_val = self.steer_slider.val
            # nur update wenn Werte sich geaendert haben
            if event is not None: 
                if self.z_curr == z_val and self.steer_curr == steer_val:
                    return
            self.z_curr = z_val
            self.steer_curr = steer_val
            self.live_movement(z_val, z_min, z_max, z_step, steer_val, F_w, F_t, *args)
            self.update_plot()
            
        self.fig.canvas.mpl_connect('button_release_event', update_on_release)
        
        # damit beim start direkt die plots da sind
        update_on_release(None)





class liveMovementStd5Link(liveMovementCommon):

    def __init__(self, kine_object):
        # Initialisierung Superklasse ausfuehren
        super().__init__(kine_object)
        # keine Rotation der Anbindungspunkte auf Chasssis-Seite
        self.phi_i = np.zeros(5)
        self.u_i = np.zeros((5, 3))
        self.omega_ai = np.zeros((5, 3))
        self.r_aiP0 = np.zeros((5, 3))



    # ==========================================
    # BLOCK 12: ERSTELLEN 3D-MATRIX FUER BETA
    # ==========================================

    def steer_vals_calc(self, z_min, z_max, z_step, steer_min, steer_max, steer_step, ai, P_aiP0, P_biP0, r_bP0):
        start_time = time.time()
        
        # fuer live_movement und zum Plotten der Kenngroessen
        self.z_array = np.flip(np.arange(z_min, z_max + z_step, z_step))
        self.steer_array = np.flip(np.arange(steer_min, steer_max + steer_step, steer_step))
        
        # Startvektor fuer beta ausgehend von der Konstruktionslage
        self.beta_start = np.concatenate((np.zeros(3), np.array(r_bP0[:2])))
        
        # zum Speichern der beta-Werte fuer live_movement
        beta_matrix_z_plus = []
        beta_matrix_z_minus = []
        
        for steer in np.arange(0, steer_max + steer_step, steer_step):
            # translatorische Verschiebung eines Anbindungspunks auf Chassis-Seite
            self.steer = steer
            # erstellt beta Liste (Matrix)
            self.z_vals_calc(z_min, z_max, z_step, ai, P_aiP0, P_biP0, r_bP0)
            # erstellt beta Matrix in positive Richtung (ergibt 3d-Matrix)
            beta_matrix_z_plus.append(self.beta_list)
            
        for steer in np.arange(-steer_step, steer_min - steer_step, -steer_step):
            # translatorische Verschiebung eines Anbindungspunks auf Chassis-Seite
            self.steer = steer
            # erstellt beta Liste (Matrix)
            self.z_vals_calc(z_min, z_max, z_step, ai, P_aiP0, P_biP0, r_bP0)
            # erstellt beta Matrix in negative Richtung (ergibt 3d-Matrix)
            beta_matrix_z_minus.append(self.beta_list)
        
        # erstellt komplette beta Matrix (3d-Matrix)
        beta_matrix_z_plus.reverse()
        self.beta_matrix = np.array(beta_matrix_z_plus + beta_matrix_z_minus)
        
        end_time = time.time()
        dauer = (end_time - start_time) * 1000
        
        # fuer die Darstellung in der Konsole
        print("\n" + "="*90)
        print(f"{'Dauer Berechnung Beta 3d-Matrix':<31} | {dauer:.3f} ms")


    def z_vals_calc(self, z_min, z_max, z_step, ai, P_aiP0, P_biP0, r_bP0):
        # Vorgabe Lenkposition
        steer_vec = np.array([0, self.steer, 0])
        r_aiDelta = np.array(ai[:5])[:, np.newaxis] * steer_vec
        
        kine.set_geometry(P_aiP0, P_biP0, self.r_aiP0, r_aiDelta, r_bP0, self.phi_i, self.u_i)
        
        # zum Speichern der beta-Werte fuer steer_vals_calc
        beta_list_z_plus = []
        beta_list_z_minus = []
        
        for z in np.arange(0, z_max + z_step, z_step):
            # Berechnung Beta
            kine.beta_calc(self.beta_start, z)
            self.beta_start = kine.beta_val
            # erstellt Beta Liste (Matrix)
            beta_list_z_plus.append(np.concatenate((self.beta_start, [z])))
            # Speichern Beta bei z=0 als Startvektor fuer naechste Lenkposition
            if z == 0:
                self.beta_start_z0 = kine.beta_val
        
        # Setzen Beta bei z=0 als Startvektor fuer negative z-Positionen
        self.beta_start = self.beta_start_z0
        
        for z in np.arange(-z_step, z_min - z_step, -z_step):
            # Berechnung Beta
            kine.beta_calc(self.beta_start, z)
            self.beta_start = kine.beta_val
            # erstellt Beta Liste (Matrix)
            beta_list_z_minus.append(np.concatenate((self.beta_start, [z])))
        
        # Setzen Beta bei z=0 als Startvektor fuer naechste Lenkposition
        self.beta_start = self.beta_start_z0
        beta_list_z_plus.reverse()
        # erstellt komplette Beta Liste (Matrix)
        self.beta_list = np.array(beta_list_z_plus + beta_list_z_minus)



    # ==========================================
    # BLOCK 13: BERECHNUNG AKTUELLE LAGE UND KENNGROESSEN
    # ==========================================

    def live_movement(self, z_curr, z_min, z_max, z_step, steer_curr, F_w, F_t, ai, P_aiP0, P_biP0, r_bP0, r, zDot, z2Dot, steerDot, steer2Dot):
        start_time = time.time()
        
        # Vorgabe Lenkposition und Lenkgeschwindigkeit
        r_vec = np.array(ai)[:, np.newaxis] * np.array([0, 1, 0])
        r_aiDelta = r_vec * steer_curr
        v_ai = r_vec * steerDot
        vDot_ai = r_vec * steer2Dot
        kine.set_geometry(P_aiP0, P_biP0, self.r_aiP0, r_aiDelta, r_bP0, self.phi_i, self.u_i)
        
        # aus der vorberechneten Beta Matrix die Beta Liste ermitteln, die der vorgegebenen Position am naechsten kommt und benutzen als Startvektoren
        # Berechnet die Differenz fuer alle, nimmt den Absolutwert und sucht den Index des Minimums
        self.steer_index = (np.abs(self.steer_array - steer_curr)).argmin()
        self.z0_index = (np.abs(self.z_array - 0)).argmin()
        beta_start = self.beta_matrix[self.steer_index, self.z0_index, :5]
        
        # zum Speichern der Kenngroessen
        data_plus = {k: [] for k in self.results.keys()}
        data_minus = {k: [] for k in self.results.keys()}

        for z in np.arange(0, z_max + z_step, z_step):
            # Berechnen Kenngrossen
            kine.calc_geometry(z, beta_start, self.omega_ai, v_ai, r)
            beta_start = kine.beta_val
            # Auslesen Kenngroessen aus self.kine.
            extracted = self.extract_kinematic_vals(z)
            for k, v in extracted.items():
                data_plus[k].append(v)
            # Speichern Beta bei z=0 als Startvektor fuer negative z-Positionen
            if z == 0:
                beta_start_z0 = kine.beta_val
        
        # Setzen Beta bei z=0 als Startvektor fuer negative z-Positionen
        beta_start = beta_start_z0
        
        for z in np.arange(-z_step, z_min - z_step, -z_step):
            # Berechnen Kenngrossen
            kine.calc_geometry(z, beta_start, self.omega_ai, v_ai, r)
            beta_start = kine.beta_val
            # Auslesen Kenngroessen aus self.kine.
            extracted = self.extract_kinematic_vals(z)
            for k, v in extracted.items():
                data_minus[k].append(v)
        
        # Speichern Kenngroessen in Listen (Matrizen)
        for k in self.results.keys():
            combined = data_plus[k][::-1] + data_minus[k]
            self.results[k] = np.array(combined)
        
        # aus der vorberechneten Beta Matrix den Beta Wert ermitteln, der der vorgegebenen Position am naechsten kommt und benutzen als Startvektor
        # Berechnet die Differenz fuer alle, nimmt den Absolutwert und sucht den Index des Minimums
        self.z_index = (np.abs(self.z_array - z_curr)).argmin()
        beta_start = self.beta_matrix[self.steer_index, self.z_index, :5]
        
        # Berechnen Position vorgegebene Lage und Kenngroessen
        kine.calc_geometry(z_curr, beta_start, self.omega_ai, v_ai, r)
        beta_val = kine.beta_val
        kine.link_force_calc(F_w, F_t)
        forces = kine.Fs
        kine.instant_centers(z_curr)
        laengspol = kine.laengspol
        querpol = kine.querpol
        
        # Berechnung Eta
        kine.eta_calc(zDot, 0, self.omega_ai, v_ai)
        eta_val = kine.eta_val
        
        # Berechnung EtaDot
        kine.etaDot_calc(zDot, z2Dot, self.omega_ai, v_ai, self.omega_ai, vDot_ai) # da self.omega_ai Nullmatrix wird self.omega_ai auch fuer omegaDot_ai benutzt
        etaDot_val = kine.etaDot_val
        
        end_time = time.time()
        dauer = (end_time - start_time) * 1000
        
        # fuer die Darstellung in der Konsole
        print("\n" + "="*90)
        print(f"{'KOMPONENTE':<17} | {'WERT'}")
        print("-" * 90)
        def fmt_arr4(arr):
            return "[" + ", ".join([f"{x:8.4e}" for x in arr]) + "]"
        def fmt_arr2(arr):
            return "[" + ", ".join([f"{x:8.2e}" for x in arr]) + "]"
        print(f"{'Beta Start':<17} | {fmt_arr4(beta_start)}")
        print(f"{'Beta Val':<17} | {fmt_arr4(beta_val)}")
        print(f"{'Eta Val':<17} | {fmt_arr4(eta_val)}")
        print(f"{'EtaDot Val':<17} | {fmt_arr4(etaDot_val)}")
        print(f"{'Lenkerkraefte':<17} | {fmt_arr2(forces)}")
        if kine.laengsTransl == 0:
            print(f"{'Laengspol':<17} | {fmt_arr4(laengspol)}")
        else:
            print(f"{'Laengspol':<17} | reine Translation, Radius unendlich")
        
        if kine.querTransl == 0:
            print(f"{'Querpol':<17} | {fmt_arr4(querpol)}")
        else:
            print(f"{'Querpol':<17} | reine Translation, Radius unendlich")
        print("-" * 90)
        print(f"{'Dauer Berechnung':<17} | {dauer:>8.2f} ms")





class liveMovementRotationAxis5Link(liveMovementCommon):

    def __init__(self, kine_object):
        # Initialisierung Superklasse ausfuehren
        super().__init__(kine_object)
        # keine Translation der Anbindungspunkte auf Chasssis-Seite
        self.r_aiDelta = np.zeros((5, 3))
        self.v_ai = np.zeros((5, 3))



    # ==========================================
    # BLOCK 14: ERSTELLEN 3D-MATRIX FUER BETA
    # ==========================================

    def steer_vals_calc(self, z_min, z_max, z_step, steer_min, steer_max, steer_step, ai, u_i, P_aiP0, P_biP0, r_aiP0, r_bP0):
        start_time = time.time()
        
        # fuer live_movement und zum Plotten der Kenngroessen
        self.z_array = np.flip(np.arange(z_min, z_max + z_step, z_step))
        self.steer_array = np.flip(np.arange(steer_min, steer_max + steer_step, steer_step))
        
        # Startvektor fuer beta ausgehend von der Konstruktionslage
        self.beta_start = np.concatenate((np.zeros(3), np.array(r_bP0[:2])))
        
        # zum Speichern der beta-Werte fuer live_movement
        beta_matrix_z_plus = []
        beta_matrix_z_minus = []
        
        for steer in np.arange(0, steer_max + steer_step, steer_step):
            # rotatorische Verschiebung eines/mehrerer Anbindungspunkte/s auf Chassis-Seite, Winkel phi
            self.steer = steer
            # erstellt beta Liste (Matrix)
            self.z_vals_calc(z_min, z_max, z_step, ai, u_i, P_aiP0, P_biP0, r_aiP0, r_bP0)
            # erstellt beta Matrix in positive Richtung (ergibt 3d-Matrix)
            beta_matrix_z_plus.append(self.beta_list)
            
        for steer in np.arange(-steer_step, steer_min - steer_step, -steer_step):
            # rotatorische Verschiebung eines/mehrerer Anbindungspunkte/s auf Chassis-Seite, Winkel phi
            self.steer = steer
            # erstellt beta Liste (Matrix)
            self.z_vals_calc(z_min, z_max, z_step, ai, u_i, P_aiP0, P_biP0, r_aiP0, r_bP0)
            # erstellt beta Matrix in positive Richtung (ergibt 3d-Matrix)
            beta_matrix_z_minus.append(self.beta_list)
        
        # erstellt komplette beta Matrix (3d-Matrix)
        beta_matrix_z_plus.reverse()
        self.beta_matrix = np.array(beta_matrix_z_plus + beta_matrix_z_minus)
        
        end_time = time.time()
        dauer = (end_time - start_time) * 1000
        
        # fuer die Darstellung in der Konsole
        print("\n" + "="*90)
        print(f"{'Dauer Berechnung Beta 3d-Matrix':<31} | {dauer:.3f} ms")


    def z_vals_calc(self, z_min, z_max, z_step, ai, u_i, P_aiP0, P_biP0, r_aiP0, r_bP0):
        # Vorgabe Lenkposition
        phi_i = np.array(ai[:5])[:, np.newaxis] * self.steer
        
        kine.set_geometry(P_aiP0, P_biP0, r_aiP0, self.r_aiDelta, r_bP0, phi_i, u_i)
        
        # zum Speichern der beta-Werte fuer steer_vals_calc
        beta_list_z_plus = []
        beta_list_z_minus = []
        
        for z in np.arange(0, z_max + z_step, z_step):
            # Berechnung Beta
            kine.beta_calc(self.beta_start, z)
            self.beta_start = kine.beta_val
            # erstellt Beta Liste (Matrix)
            beta_list_z_plus.append(np.concatenate((self.beta_start, [z])))
            # Speichern Beta bei z=0 als Startvektor fuer naechste Lenkposition
            if z == 0:
                self.beta_start_z0 = kine.beta_val
        
        # Setzen Beta bei z=0 als Startvektor fuer negative z-Positionen
        self.beta_start = self.beta_start_z0
        
        for z in np.arange(-z_step, z_min - z_step, -z_step):
            # Berechnung Beta
            kine.beta_calc(self.beta_start, z)
            self.beta_start = kine.beta_val
            # erstellt Beta Liste (Matrix)
            beta_list_z_minus.append(np.concatenate((self.beta_start, [z])))
        
        # Setzen Beta bei z=0 als Startvektor fuer naechste Lenkposition
        self.beta_start = self.beta_start_z0
        beta_list_z_plus.reverse()
        # erstellt komplette Beta Liste (Matrix)
        self.beta_list = np.array(beta_list_z_plus + beta_list_z_minus)



    # ==========================================
    # BLOCK 15: BERECHNUNG AKTUELLE LAGE UND KENNGROESSEN
    # ==========================================

    def live_movement(self, z_curr, z_min, z_max, z_step, steer_curr, F_w, F_t, ai, P_aiP0, P_biP0, r_aiP0, r_bP0, r, u_i, zDot, z2Dot, steerDot, steer2Dot):
        start_time = time.time()
        
        # Vorgabe Lenkposition und Lenkgeschwindigkeit
        phi_i = np.array(ai)[:, np.newaxis] * steer_curr
        
        omega_vec = np.array(u_i) * np.array(ai)[:, np.newaxis]
        omega_ai = omega_vec * steerDot
        omegaDot_ai = omega_vec * steer2Dot
        
        
        kine.set_geometry(P_aiP0, P_biP0, r_aiP0, self.r_aiDelta, r_bP0, phi_i, u_i)
        
        # aus der vorberechneten Beta Matrix die Beta Liste ermitteln, die der vorgegebenen Position am naechsten kommt und benutzen als Startvektoren
        # Berechnet die Differenz fuer alle, nimmt den Absolutwert und sucht den Index des Minimums
        self.steer_index = (np.abs(self.steer_array - steer_curr)).argmin()
        self.z0_index = (np.abs(self.z_array - 0)).argmin()
        beta_start = self.beta_matrix[self.steer_index, self.z0_index, :5]
        
        # zum Speichern der Kenngroessen
        data_plus = {k: [] for k in self.results.keys()}
        data_minus = {k: [] for k in self.results.keys()}

        for z in np.arange(0, z_max + z_step, z_step):
            # Berechnen Kenngrossen
            kine.calc_geometry(z, beta_start, omega_ai, self.v_ai, r)
            beta_start = kine.beta_val
            # Auslesen Kenngroessen aus self.kine.
            extracted = self.extract_kinematic_vals(z)
            for k, v in extracted.items():
                data_plus[k].append(v)
            # Speichern Beta bei z=0 als Startvektor fuer negative z-Positionen
            if z == 0:
                beta_start_z0 = kine.beta_val
        
        # Setzen Beta bei z=0 als Startvektor fuer negative z-Positionen
        beta_start = beta_start_z0
        
        for z in np.arange(-z_step, z_min - z_step, -z_step):
            # Berechnen Kenngrossen
            kine.calc_geometry(z, beta_start, omega_ai, self.v_ai, r)
            beta_start = kine.beta_val
            # Auslesen Kenngroessen aus self.kine.
            extracted = self.extract_kinematic_vals(z)
            for k, v in extracted.items():
                data_minus[k].append(v)
        
        # Speichern Kenngroessen in Listen (Matrizen)
        for k in self.results.keys():
            combined = data_plus[k][::-1] + data_minus[k]
            self.results[k] = np.array(combined)
        
        # aus der vorberechneten Beta Matrix den Beta Wert ermitteln, der der vorgegebenen Position am naechsten kommt und benutzen als Startvektor
        # Berechnet die Differenz fuer alle, nimmt den Absolutwert und sucht den Index des Minimums
        self.z_index = (np.abs(self.z_array - z_curr)).argmin()
        beta_start = self.beta_matrix[self.steer_index, self.z_index, :5]
        
        # Berechnen Position vorgegebene Lage und Kenngroessen
        kine.calc_geometry(z_curr, beta_start, omega_ai, self.v_ai, r)
        beta_val = kine.beta_val
        kine.link_force_calc(F_w, F_t)
        forces = kine.Fs
        kine.instant_centers(z_curr)
        laengspol = kine.laengspol
        querpol = kine.querpol
        
        # Berechnung Eta
        kine.eta_calc(zDot, 0, omega_ai, self.v_ai)
        eta_val = kine.eta_val
        
        # Berechnung EtaDot
        kine.etaDot_calc(zDot, z2Dot, omega_ai, self.v_ai, omegaDot_ai, self.v_ai) # da self.v_ai Nullmatrix wird self.v_ai auch fuer vDot_ai benutzt
        etaDot_val = kine.etaDot_val
        
        end_time = time.time()
        dauer = (end_time - start_time) * 1000
        
        # fuer die Darstellung in der Konsole
        print("\n" + "="*90)
        print(f"{'KOMPONENTE':<17} | {'WERT'}")
        print("-" * 90)
        def fmt_arr4(arr):
            return "[" + ", ".join([f"{x:8.4e}" for x in arr]) + "]"
        def fmt_arr2(arr):
            return "[" + ", ".join([f"{x:8.2e}" for x in arr]) + "]"
        print(f"{'Beta Start':<17} | {fmt_arr4(beta_start)}")
        print(f"{'Beta Val':<17} | {fmt_arr4(beta_val)}")
        print(f"{'Eta Val':<17} | {fmt_arr4(eta_val)}")
        print(f"{'EtaDot Val':<17} | {fmt_arr4(etaDot_val)}")
        print(f"{'Lenkerkraefte':<17} | {fmt_arr2(forces)}")
        if kine.laengsTransl == 0:
            print(f"{'Laengspol':<17} | {fmt_arr4(laengspol)}")
        else:
            print(f"{'Laengspol':<17} | reine Translation, Radius unendlich")
        if kine.querTransl == 0:
            print(f"{'Querpol':<17} | {fmt_arr4(querpol)}")
        else:
            print(f"{'Querpol':<17} | reine Translation, Radius unendlich")
        print("-" * 90)
        print(f"{'Dauer Berechnung':<17} | {dauer:>8.2f} ms")











# ==========================================
# BLOCK 16: HAUPTPROGRAMM (VOLLE SIMULATION)
# ==========================================

kine = kinematics()

# fuer Std Axis
simulation_STD = liveMovementStd5Link(kine)

# fuer Rot Axis
simulation_ROT = liveMovementRotationAxis5Link(kine)

if __name__ == "__main__":
    
    if config == 1:
        # fuer Rot Axis
        simulation_ROT.steer_vals_calc(z_min, z_max, z_step, steer_min_ROT, steer_max_ROT, steer_step_ROT, ai_ROT, u_i, P_aiP0_ROT, P_biP0_ROT, r_aiP0_ROT, r_bP0_ROT)
        simulation_ROT.setup_plot(P_aiP0_ROT, P_biP0_ROT, r_bP0_ROT, r)
        simulation_ROT.add_sliders(z_min, z_max, z_step, steer_min_ROT, steer_max_ROT, steer_step_ROT, F_w, F_t, ai_ROT, P_aiP0_ROT, P_biP0_ROT, r_aiP0_ROT, r_bP0_ROT, r, u_i, zDot, z2Dot, steerDot_ROT, steer2Dot_ROT)
    
    if config == 0:
        # fuer Std Axis
        simulation_STD.steer_vals_calc(z_min, z_max, z_step, steer_min_STD, steer_max_STD, steer_step_STD, ai_STD, P_aiP0_STD, P_biP0_STD, r_bP0_STD)
        simulation_STD.setup_plot(P_aiP0_STD, P_biP0_STD, r_bP0_STD, r)
        simulation_STD.add_sliders(z_min, z_max, z_step, steer_min_STD, steer_max_STD, steer_step_STD, F_w, F_t, ai_STD, P_aiP0_STD, P_biP0_STD, r_bP0_STD, r, zDot, z2Dot, steerDot_STD, steer2Dot_STD)
    
    plt.show(block=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    