import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from numba.typed import List

#Definition des constantes physique
M = 1/2 #Masse du trou noir
Rs = 2*M #Rayon de Schwarzschild

epaisseur_disque = 0.3 #Epaisseur du disque selon z
rayon_disque = 10*Rs #Rayon maximal du disque d'accretion
alpha = 1e-3 #Paramètre de viscosité

#Initalisation des paramètres de l'image
#coordonnees de la camera
Y_camera = -10
Z_camera = 0

#Resolution de l'image
nombre_de_pixel_x = 512
nombre_de_pixel_z = 512


taille_du_pixel = 0.024*2 #Largeur physique de chaque pixel #0.024 pour 1024 doit marcher
#0.0123 pour 2048
taille_selon_x = nombre_de_pixel_x * taille_du_pixel #largeur physique selon x
taille_selon_z = nombre_de_pixel_z * taille_du_pixel #largeur physique selon z

#Initialisation des coordonnes initiales
coordonnees_initiales_photons_x = np.linspace(-taille_selon_x/2, taille_selon_x/2, nombre_de_pixel_x)
coordonnees_initiales_photons_z = np.linspace(-taille_selon_z/2 + Z_camera, taille_selon_z/2 +Z_camera, nombre_de_pixel_z)

dphi = 1e-2 #pas angulaire

#Fonctions utilisées pour le calcul de la trajectoire
@njit
def remplir_r(r, i, b):
    r_plus = r[i] - (r[i]**2) * dphi * np.sqrt((1/(b**2)) - (1/(r[i]**2))*(1- ((2*M)/(r[i]))))
    if r_plus <= Rs:
        return -2
    if r_plus <= 0 or np.isnan(r_plus): #on teste si on a atteint la valeur limite dr/dphi = 0
        return -1
    else:
        return r_plus

@njit
def convertir_polaires_en_cartesiennes(r, phi):
    return (r* np.cos(phi), r*np.sin(phi))

def r_trajectoire_complete(tableau):
    n = len(tableau)
    for i in range(n):
        tableau.append(tableau[n-i-1])
    return tableau

def phi_trajectoire_complete(tableau):
    for j in range(len(tableau)):
        tableau.append(tableau[-1] + dphi)
    return tableau

def trajectoire_pour_un_photon(phi_initial, x_initial):
    r_initial = x_initial/np.cos(phi_initial) #distance radiale initiale
    b = np.sin(phi_initial) * r_initial #paramètre d'impact
    phi_negatif = (phi_initial < 0)

    tableau_phi = [np.abs(phi_initial)]
    tableau_r = List()
    tableau_r.append(r_initial)
    tableau_x = [x_initial]
    tableau_y = [b]

    i = 0
    horizon_atteint = False
    while remplir_r(tableau_r, i, b) != -1:
        if remplir_r(tableau_r, i, b) == -2:
            horizon_atteint = True
            break
        tableau_phi.append(tableau_phi[i] + dphi)
        tableau_r.append(remplir_r(tableau_r, i, b))
        i +=1

    if not horizon_atteint:
        tableau_r = r_trajectoire_complete(tableau_r)
        tableau_phi = phi_trajectoire_complete(tableau_phi)
    if phi_negatif:
        for i in range(len(tableau_r)):
            X, Y = convertir_polaires_en_cartesiennes(tableau_r[i], tableau_phi[i])
            tableau_x.append(X)
            tableau_y.append(-Y)
    else:
        for i in range(len(tableau_r)):
            X, Y = convertir_polaires_en_cartesiennes(tableau_r[i], tableau_phi[i])
            tableau_x.append(X)
            tableau_y.append(Y)

    return tableau_x, tableau_y

def convertir_coordonnes_planaires_3D(tableau_x, tableau_y, angle_au_pixel):
    return np.array(tableau_y)*np.sin(angle_au_pixel), np.array(tableau_x), np.array(tableau_y)*np.cos(angle_au_pixel)

def couleur_pixel(X, Y, Z):
    for i in range(len(Z)):
        r = np.sqrt(X[i]**2 + Y[i]**2 + Z[i]**2)
        if r < 6/5*Rs:
            return (0, 0, 0)
        if np.abs(Z[i]) < epaisseur_disque/2 and r < rayon_disque and r > 3*Rs:
            T = temperature_disque(alpha, r)
            return temperature_vers_RGB(luminance_corps_noir, decalage_Doppler(X[i], Y[i], r)*decalage_Doppler_gravitationnel(r) *T) #pour tous les effets :
    return (0.0, 0.0, 0.0)

def luminance_corps_noir(lambda_, T):
    c1 = 3.718e-16
    c2 = 1.439e-2
    return (c1 * (1/(lambda_**5)) * 1/(np.exp(c2/(lambda_*T))-1))

def temperature_disque(alpha, r):
    return 3000* (r/(3*Rs))**(-3/4) #ici, on prend M = 1/2 masse stellaires, le coefficient devant est en réalité 2.3e7 # préfacteur pour disque alpha = 4.6e2 *(alpha*M)**(-1/4)

def decalage_Doppler(x, y, r):
    v = np.sqrt(M/r) #vitesse au rayon r
    vy = v * np.cos(np.arctan2(y, x))
    return np.sqrt((1 - vy)/(1+vy))

def decalage_Doppler_gravitationnel(r):
    r_prime = np.abs(Y_camera) #distance radiale de l'observateur
    return np.sqrt((1-Rs/r_prime)/(1-Rs/r))


def temperature_vers_RGB(luminance_corps_noir, T):
    #table de conversion qui permet de faire le lien entre la longueur d'onde et le système de couleur X, Y, Z CIE
    table_de_conversion_CIE = [[0.0014,0.0000,0.0065], [0.0022,0.0001,0.0105], [0.0042,0.0001,0.0201],[0.0076,0.0002,0.0362], [0.0143,0.0004,0.0679], [0.0232,0.0006,0.1102], [0.0435,0.0012,0.2074], [0.0776,0.0022,0.3713], [0.1344,0.0040,0.6456],[0.2148,0.0073,1.0391], [0.2839,0.0116,1.3856],[0.3285,0.0168,1.6230],[0.3483,0.0230,1.7471], [0.3481,0.0298,1.7826], [0.3362,0.0380,1.7721],[0.3187,0.0480,1.7441], [0.2908,0.0600,1.6692], [0.2511,0.0739,1.5281],[0.1954,0.0910,1.2876], [0.1421,0.1126,1.0419], [0.0956,0.1390,0.8130],[0.0580,0.1693,0.6162], [0.0320,0.2080,0.4652], [0.0147,0.2586,0.3533],[0.0049,0.3230,0.2720], [0.0024,0.4073,0.2123], [0.0093,0.5030,0.1582],[0.0291,0.6082,0.1117], [0.0633,0.7100,0.0782], [0.1096,0.7932,0.0573],[0.1655,0.8620,0.0422], [0.2257,0.9149,0.0298], [0.2904,0.9540,0.0203],[0.3597,0.9803,0.0134], [0.4334,0.9950,0.0087], [0.5121,1.0000,0.0057],[0.5945,0.9950,0.0039], [0.6784,0.9786,0.0027], [0.7621,0.9520,0.0021],[0.8425,0.9154,0.0018], [0.9163,0.8700,0.0017], [0.9786,0.8163,0.0014],[1.0263,0.7570,0.0011], [1.0567,0.6949,0.0010], [1.0622,0.6310,0.0008],[1.0456,0.5668,0.0006], [1.0026,0.5030,0.0003], [0.9384,0.4412,0.0002],[0.8544,0.3810,0.0002], [0.7514,0.3210,0.0001], [0.6424,0.2650,0.0000],[0.5419,0.2170,0.0000], [0.4479,0.1750,0.0000], [0.3608,0.1382,0.0000],[0.2835,0.1070,0.0000],
    [0.2187,0.0816,0.0000], [0.1649,0.0610,0.0000],[0.1212,0.0446,0.0000], [0.0874,0.0320,0.0000], [0.0636,0.0232,0.0000],[0.0468,0.0170,0.0000], [0.0329,0.0119,0.0000], [0.0227,0.0082,0.0000],[0.0158,0.0057,0.0000], [0.0114,0.0041,0.0000], [0.0081,0.0029,0.0000],[0.0058,0.0021,0.0000], [0.0041,0.0015,0.0000],[0.0029,0.0010,0.0000],[0.0020,0.0007,0.0000], [0.0014,0.0005,0.0000], [0.0010,0.0004,0.0000],[0.0007,0.0002,0.0000], [0.0005,0.0002,0.0000], [0.0003,0.0001,0.0000],[0.0002,0.0001,0.0000], [0.0002,0.0001,0.0000], [0.0001,0.0000,0.0000],[0.0001,0.0000,0.0000], [0.0001,0.0000,0.0000], [0.0000,0.0000,0.0000]]
    lambda_0 = 380e-9
    X = 0
    Y = 0
    Z = 0
    for i in range(81):
        intensite = luminance_corps_noir(lambda_0, T)
        X += intensite *table_de_conversion_CIE[i][0]
        Y += intensite * table_de_conversion_CIE[i][1]
        Z += intensite *table_de_conversion_CIE[i][2]
        lambda_0 += 5e-9
    somme = X + Y + Z
    xyz = (X/somme, Y/somme, Z/somme)

    #Matrice de conversion entre les coordonnees X, Y, Z et les valeurs RGB selon les coordonnees standard RGB, espace de couleur par défaut utilisé numériquement
    matrice_de_conversion = np.array([[3.2404542, -1.5371385, -0.4985314],[-0.9692660, 1.8760108, 0.0415560],[0.0556434, -0.2040259, 1.0572252]])
    RGB = np.dot(matrice_de_conversion, np.array(xyz))
    RGB_correction_alpha = [12.92*C if C < 0.00313 else (1+ 0.055)*(C**(1/3)) - 0.055 for C in RGB] #On corrige le gamma avec gamma = 3
    return RGB_correction_alpha[0], RGB_correction_alpha[1], RGB_correction_alpha[2]

#Calcul des trajectoires et des couleurs pour tous les photons de l'image
image = [[(0, 0, 0)]*nombre_de_pixel_x for h in range(nombre_de_pixel_z)]
for indice_z in range(nombre_de_pixel_z):
    print("ligne no" + str(indice_z))
    for indice_x in range(nombre_de_pixel_x):
        b = np.sqrt(coordonnees_initiales_photons_x[indice_x]**2 + coordonnees_initiales_photons_z[indice_z]**2)
        r_initial = np.sqrt(coordonnees_initiales_photons_x[indice_x]**2 + coordonnees_initiales_photons_z[indice_z]**2 + Y_camera**2)
        phi_initial = np.arcsin(b/r_initial)
        x, y = trajectoire_pour_un_photon(phi_initial, np.abs(Y_camera))
        X, Y, Z = convertir_coordonnes_planaires_3D(x, y, np.arctan2(coordonnees_initiales_photons_x[indice_x], coordonnees_initiales_photons_z[indice_z]))
        image[indice_z][indice_x] = couleur_pixel(X, Y, Z)

image[0][0] = (1, 1, 1)
plt.imshow(image, interpolation = None)
plt.show()
