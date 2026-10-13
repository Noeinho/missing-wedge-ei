import torch
import math
from torch.fft import fft2, ifft2

def wedge_mask(n: int, half_angle_deg:float) -> torch.Tensor:
    """Masque de Fourier du missing wedge. Masque ie des 1 la ou on voit, des 0 la ou la freq est manquante.
    n : taille de l'image (n x n)
    half_angle_deg : demi-angle du cône en degrés (angle total = 2 * half_angle_deg)
    """
    k = torch.fft.fftfreq(n) #donne les n fréquences k/n avec convention centrée en 0
    kx, ky = torch.meshgrid(k, k, indexing="xy") #on crée une grille de fréquences (kx, ky) de shape (n, n).
    
    #Il faut maintenant créer le masque. Le masque vaut 1 quand on est à l'extérieur du cône.
    # Donc le masque vaut 1. quand |kx| > |ky| * tan(half_angle)
    # Créons un masque booléen avec cette condition.
    mask = torch.abs(kx) > torch.abs(ky) * math.tan(math.radians(half_angle_deg))
    #rq en torch on peut faire du booléen, ça fera la comparaison elt par elt.
    #dépend de la convention de l'angle. kx et ky, à un facteur Fs près, sont les fréquences réelles. le facteur ne change rien.

    mask = mask.to(torch.float32)
    mask[0, 0] = 1.0   # on garde la fréquence nulle (moyenne de l'image)
    return mask

def measure(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Applique l'opérateur A, ie supprime les fréquences masquées.
    x: (B, C, N, N) : batch d'images
    mask: (N, N) : masque de Fourier du missing wedge"""
    return ifft2(mask * fft2(x)).real #équivaut a une convolution dans l'espace spatial. 

if __name__ == "__main__":
    print(wedge_mask(4, 30.))      # seulement si on lance forward_op.py directement, pas si on l'importe.