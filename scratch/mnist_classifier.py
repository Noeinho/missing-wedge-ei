import torch
#Torch fait de la différenciation automatique c'est à dire qu'il dérive en utilisant les formules de dérivations exactes.
#Il ne fait pas d'estimation numérique de gradient.

x = torch.tensor([1., 2., 3.], requires_grad = True) #tensor = tableau multidimensionnel torch
#requires_grad demande à PyTorch de suivre les informations de ce tenseur pour calculer des futurs gradients wrt x.
#I.e x est une variable de laquelle on voudra calculer le gradient de la perte.

y = (x**2).sum() #x² est appliqué élément par élément, puis sum() additionne tous les éléments du tenseur résultant.
#Le résultat est un tenseur scalaire (0d, un seul nombre), tensor(14.).

y.backward() #calcul du gradient par backpropagation. Dérive y par rapport aux tenseurs suivis (ici x.) Torch a suivi que
#y = x² et donc peut calculer dy/dx = 2x. Le gradient évalué en x est stocké dans x.grad.

print(x.grad)  # Output: tensor([2., 4., 6.])

z = (x**2) #si notre fonction n'est pas scalaire, on ne peut pas utiliser backward sans argument.

#Il faut spécifier le gradient de sortie, c'est à dire le gradient de la fonction désirée par rapport à z.
#Par exemple si on veut calculer le gradient de 2z1 par rapport à x, le gradient de sortie est d(2z1)/dz = [2, 0, 0]

z.backward(torch.tensor([2., 0., 0.]))

#Attention faire un backward oublie le graphe de calcul entre z et x. On ne peut donc pas refaire un backward sur z atm.

#L'idée est qu'en fait on a pas besoin de nous s'embêter à calculer le gradient de sortie.
#On peut juste dire qu'on veut le gradient de la loss L = 2z1 wrt x, et faire backward() sur L.
z = (x**2)
L = 2*z[0] #L = 2z1 = 2x1²
L.backward() #calcul du gradient de L par rapport à x.

print(x.grad)
#Attention, le gradient est accumulé dans x.grad (i.e sommé entre toutes les fonctions de x dérivées.)

x.grad.zero_() #On peut réinitialiser le gradient à zéro pour ne pas accumuler les gradients. 
#Attention nos bacwkard ont fait oublier le graphe de calcul. Il faut redéfinir z et L pour refaire un backward.
z = (x**2)
L = 2*z[0] 
L.backward()
print(x.grad)


import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Charger le dataset MNIST
dataset = torchvision.datasets.MNIST(
    root="data", #stocker les données télécharger dans le dossier data.
    train=True, #charge la partie Train de MNIST
    download=True, #telecharge le dataset si pas trouvé dans data
    transform=ToTensor() #transform permet de spécifier une transformation a appliquer a chaque image récupérée.
    #ici on les transformes en tenseur torch de shape (1, 28, 28) ou les valeurs sont les pixels entre 0 et 1.
    #Le 1 dit qu'on travaille en niveaux de gris donc on a besoin que d'un chiffre pour représenter un pixel.
    #En RGB, ç'aurait été 3.
)

# 2. Créer le DataLoader
loader = DataLoader(
    dataset, #sur notre Dataset
    batch_size=64, #répartis le en batch de taille 64. Nouvelle dimension : (64, 1, 28, 28). #labels de taille (64,)
    shuffle=True #en mélangeant les données.
)
#l'objet loader est fait pour être itéré dessus (itérer sur les batchs créés)

# 3. Récupérer un batch
images, labels = next(iter(loader))

# 4. Afficher les formes
print("Forme des images :", images.shape)
print("Forme des labels :", labels.shape)

# 5. Afficher une image du batch
image = images[0]  # Première image du batch
label = labels[0].item()

plt.imshow(image.squeeze(0), cmap="gray")
plt.title(f"Label : {label}")
plt.axis("off")
plt.show()


import torch.nn as nn
import torch.optim as optim

# Choisir le matériel de calcul
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Définir le modèle
class MNISTClassifier(nn.Module): #standard de définir une classe
    def __init__(self):
        super().__init__()

        # Deux couches de convolution
        self.conv1 = nn.Conv2d(
            in_channels=1, #niveau de gris donc 1 canal d'entrée
            out_channels=16, #on va apprendre 16 filtres de convolution qui donnent 16 features map.
            kernel_size=3, #chaque filtre est 3x3
            padding=1 #rajoute une bordure de 1 pixel autour de l'image, pour que la taille des features map en sortie
            #soit la meme que celle de l'image d'entrée. (28X28). Stride = 1.
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Fonction d'activation
        self.relu = nn.ReLU()

        # Réduire la taille spatiale des images
        self.pool = nn.MaxPool2d(kernel_size=2) #conserve la valeur max dans chaque fenetre 2X2. 
        #Utile pour réduire la taille des features maps en gardant l'activation la plus puissante.
        #Ici la taille est divisée par 2 en largeur et en hauteur.

        # Une couche linéaire pour les 10 classes, fc pour fully connected.
        self.fc = nn.Linear(32 * 7 * 7, 10) #7 car apres 2 maxpooling, on a divisé par 4 la taille des maps.

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # Aplatir les caractéristiques avant la couche linéaire
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x) #on met pas de couche softmax car la crossentropy le fait pour nous (elle attend les logits)
        return x


# 2. Créer le modèle
model = MNISTClassifier().to(device) #initialisation des poids, et envoi sur le device.

# 3. Définir la fonction de perte
criterion = nn.CrossEntropyLoss()

# 4. Définir l'optimiseur
optimizer = optim.Adam(model.parameters(), lr=0.001) #parametres retrouvés automatiquement car usage de nn.Module


# 5. Boucle d'entraînement
num_epochs = 5

for epoch in range(num_epochs):
    model.train() #modèle placé en mode entrainement. (utile pour certaines couches, pas vraiment ici). Attention n'entraine pas.

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Réinitialiser les gradients
        optimizer.zero_grad()

        # Propagation avant
        outputs = model(images) #Les logits donnés par le modèle de shape (64, 10)

        # Calcul de la perte
        loss = criterion(outputs, labels) #labels de taille (64,). Ce sont les tailles attendues par CE loss.

        # Rétropropagation
        loss.backward() #comme vu au dessus, la sortie étant scalaire, la rétropropagation est automatique.

        # Mise à jour des paramètres
        optimizer.step()

        # Statistiques
        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Loss: {total_loss / total_samples:.4f} | "
        f"Accuracy: {100 * total_correct / total_samples:.2f}%"
    )


# 1. Charger le dataset MNIST
test_dataset = torchvision.datasets.MNIST(
    root="data", #stocker les données télécharger dans le dossier data.
    train=False, #charge la partie Test de MNIST
    download=True, #telecharge le dataset si pas trouvé dans data
    transform=ToTensor() #transform permet de spécifier une transformation a appliquer a chaque image récupérée.
    #ici on les transformes en tenseur torch de shape (1, 28, 28) ou les valeurs sont les pixels entre 0 et 1.
    #Le 1 dit qu'on travaille en niveaux de gris donc on a besoin que d'un chiffre pour représenter un pixel.
    #En RGB, ç'aurait été 3.
)

# 2. Créer le DataLoader
test_loader = DataLoader(
    test_dataset, #sur notre Dataset
    batch_size=64, #répartis le en batch de taille 64. Nouvelle dimension : (64, 1, 28, 28). #labels de taille (64,)
    shuffle=False #en mélangeant les données.
)
#l'objet loader est fait pour être itéré dessus (itérer sur les batchs créés)

# Mettre le modèle en mode évaluation
model.eval()

# Compteurs
total_correct = 0
total_samples = 0

# Désactiver le calcul des gradients
with torch.no_grad(): #désactive le suivi des gradients. Puisqu'on ne fait plus de backward, on n'a pas besoin.
    #Economise donc de la mémoire et accélère les calculs.

    for images, labels in test_loader:

        # Envoyer les données sur le même appareil que le modèle
        images = images.to(device)
        labels = labels.to(device)

        # Faire les prédictions
        outputs = model(images)

        # Récupérer la classe prédite pour chaque image
        predictions = outputs.argmax(dim=1) #l'argmax sur la dim 1 c'est à dire sur les scores.

        # Compter les prédictions correctes
        total_correct += (predictions == labels).sum().item() #.sum() car predictions et labels sont de taille 64

        # Compter le nombre total d'images
        total_samples += labels.size(0)

# Calculer l'accuracy en pourcentage
accuracy = 100 * total_correct / total_samples

print(f"Accuracy sur les données de test : {accuracy:.2f}%")
