import argparse
import os
import time
import scipy.stats as st
import wandb
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import higher
import matplotlib.pyplot as plt

import layers_2425_AP as layers
from inner_optimizers import InnerOptBuilder
from rotated_mnist_main.rotated_mnist import flattened_rotMNIST

OUTPUT_PATH = "./outputs/rotated_mnist_outputs"


def load_rmnist_task_data(loader, num_tasks, k_spt, k_qry):
    """Transformation des données pour créer les ensembles support et requête"""
    task_data = []
    for batch_idx, (images, labels, angles) in enumerate(loader):
        x_spt, y_spt, angles_spt = images[:k_spt], labels[:k_spt], angles[:k_spt]
        x_qry, y_qry, angles_qry = images[k_spt:k_spt + k_qry], labels[k_spt:k_spt + k_qry], angles[k_spt:k_spt + k_qry]

        task_data.append((x_spt, y_spt, angles_spt, x_qry, y_qry, angles_qry))
        if len(task_data) >= num_tasks:
            break

    return task_data

def visualize_weights(net, architecture, step_idx):
    """Visualise et log les poids du modèle dans WandB"""
    with torch.no_grad():
        if architecture == "shared":
            # Cas où l'architecture est partagée
            shared_layer = net[0]  # La première couche partagée
            weights = shared_layer.weight.cpu().numpy()  # Poids des filtres convolutifs
            
            # Créer une figure avec les filtres
            num_filters = weights.shape[0]
            fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
            for i, ax in enumerate(axes):
                ax.imshow(weights[i, 0], cmap="viridis")  # Poids du filtre
                ax.axis("off")
            plt.suptitle(f"Weights Visualization (Shared Architecture) - Step {step_idx}", fontsize=14)
        elif architecture == "dedicated":
            # Cas où il y a des circuits dédiés
            circuits = net.circuits
            fig, axes = plt.subplots(len(circuits), 1, figsize=(10, len(circuits) * 3))
            for i, circuit in enumerate(circuits):
                dedicated_weights = circuit[0].weight.cpu().numpy()
                ax = axes[i]
                ax.imshow(dedicated_weights[0, 0], cmap="viridis")
                ax.axis("off")
                ax.set_title(f"Circuit {i + 1} Weights")
            plt.suptitle(f"Weights Visualization (Dedicated Architecture) - Step {step_idx}", fontsize=14)
        else:
            raise ValueError("Architecture non reconnue pour la visualisation des poids.")

        # Log directement dans WandB
        wandb.log({f"weights_visualization_step_{step_idx}": wandb.Image(fig)}, step=step_idx)
        plt.close(fig)



def visualize_activations(net, x_sample, architecture, step_idx):
    """Visualise et log les activations intermédiaires du modèle dans WandB"""
    with torch.no_grad():
        if architecture == "shared":
            activations = []
            current_input = x_sample
            for layer in net:
                current_input = layer(current_input)
                if isinstance(layer, nn.ReLU):  # Prendre les activations après chaque ReLU
                    activations.append(current_input.cpu().numpy())

            # Créer un graphique pour chaque couche
            num_activations = len(activations)
            fig, axes = plt.subplots(1, num_activations, figsize=(15, 5))

            # Gérer le cas où il n'y a qu'une seule activation
            if num_activations == 1:
                axes = [axes]

            for i, activation in enumerate(activations):
                # Réduire les dimensions pour rendre compatible avec imshow
                # Prendre la moyenne sur les canaux pour obtenir une image 2D
                activation_2d = activation[0].mean(axis=0)  # Moyenne sur les canaux

                ax = axes[i]
                ax.imshow(activation_2d, cmap="viridis", aspect="auto")
                ax.axis("off")
                ax.set_title(f"Layer {i + 1}")
            plt.suptitle(f"Activations Visualization (Shared Architecture) - Step {step_idx}", fontsize=14)

        elif architecture == "dedicated":
            circuits = net.circuits
            num_circuits = len(circuits)
            fig, axes = plt.subplots(num_circuits, 1, figsize=(10, num_circuits * 3))

            # Gérer le cas où il n'y a qu'un seul circuit
            if num_circuits == 1:
                axes = [axes]

            for i, circuit in enumerate(circuits):
                activations = circuit(x_sample).cpu().numpy()

                # Réduire les dimensions pour rendre compatible avec imshow
                activation_2d = activations[0].mean(axis=0)  # Moyenne sur les canaux

                ax = axes[i]
                ax.imshow(activation_2d, cmap="viridis", aspect="auto")
                ax.axis("off")
                ax.set_title(f"Activations from Circuit {i + 1}")
            plt.suptitle(f"Activations Visualization (Dedicated Architecture) - Step {step_idx}", fontsize=14)
        else:
            raise ValueError("Architecture non reconnue pour la visualisation des activations.")

        # Log directement dans WandB
        wandb.log({f"activations_visualization_step_{step_idx}": wandb.Image(fig)}, step=step_idx)
        plt.close(fig)


def train(step_idx, task_data, net, inner_opt_builder, meta_opt, n_inner_iter, architecture, weight):
    """Main meta-training step for RMNIST."""
    qry_losses = []
    meta_opt.zero_grad()
    angle_errors = {}  # Dictionnaire pour stocker les erreurs par angle
    angle_counts = {}  # Dictionnaire pour stocker le nombre d'images par angle

    for task in task_data:
        x_spt, y_spt, angles_spt, x_qry, y_qry, angles_qry = task
        
        task_num = x_spt.size(0)
        inner_opt = inner_opt_builder.inner_opt

        # Copie du bout de test() pour compter les images par angle
        for angle in angles_qry:
            angle_val = angle.item()
            if angle_val not in angle_errors:
                angle_errors[angle_val] = 0
                angle_counts[angle_val] = 0

        with higher.innerloop_ctx(
            net,
            inner_opt,
            copy_initial_weights=False,
            override=inner_opt_builder.overrides,
        ) as (fnet, diffopt):
            # Inner-loop updates on the support set
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt)
                spt_loss = F.cross_entropy(spt_pred, y_spt)  # Classification loss
                diffopt.step(spt_loss)

            # Query set evaluation
            qry_pred = fnet(x_qry)
            qry_loss = F.cross_entropy(qry_pred, y_qry)
            qry_losses.append(qry_loss.detach().cpu().numpy())
            qry_loss.backward()

            # Comptage des images par angle
            qry_pred_classes = torch.argmax(qry_pred, dim=1)  # Prédictions des classes
            for pred, label, angle in zip(qry_pred_classes, y_qry, angles_qry):
                angle_val = angle.item()
                angle_counts[angle_val] += 1  # Incrémenter le compteur d'images

    # /!\ Pour visualiser le weight sharing de train, ne pas mettre en même temps que le test.
    # **Appel à visualize_weights et visualize_activations**
    # if weight and (step_idx == 0 or (step_idx + 1) % 100 == 0):
    #     # Visualiser les poids appris
    #     visualize_weights(net, architecture, step_idx)

    #     # Visualiser les activations intermédiaires pour un échantillon (query set)
    #     x_sample = task_data[0][3][:1]  # Prendre un échantillon du query set
    #     visualize_activations(net, x_sample, architecture, step_idx)

    # Générer le graphique en barres pour le nombre d'images par angle
    angles = list(angle_counts.keys())
    counts = list(angle_counts.values())
    plt.figure(figsize=(10, 6))
    plt.bar(angles, counts, color='skyblue')
    plt.xlabel("Angle de Rotation (°)")
    plt.ylabel("Nombre d'Images")
    plt.title("Nombre d'Images par Angle de Rotation (TRAIN)")
    plt.xticks(angles)
    plt.tight_layout()
    # Log du graphique directement dans wandb
    wandb.log({"angle_counts_barplot1": wandb.Image(plt)}, step=step_idx)
    plt.close()

    # Meta-optimization step
    metrics = {"train_loss": np.mean(qry_losses)}
    wandb.log(metrics, step=step_idx)
    meta_opt.step()

def test(step_idx, task_data, net, inner_opt_builder, n_inner_iter, architecture, weight):
    """Main meta-testing step for RMNIST"""
    qry_losses = []
    angle_errors = {}  # Dictionnaire pour stocker les erreurs par angle
    angle_counts = {}  # Dictionnaire pour stocker le nombre d'images par angle

    for task in task_data:
        x_spt, y_spt, angles_spt, x_qry, y_qry, angles_qry = task

        task_num = x_spt.size(0)
        inner_opt = inner_opt_builder.inner_opt

        # Initialisation des erreurs par angle
        for angle in angles_qry:
            angle_val = angle.item()
            if angle_val not in angle_errors:
                angle_errors[angle_val] = 0
                angle_counts[angle_val] = 0

        with higher.innerloop_ctx(
            net, inner_opt, track_higher_grads=False, override=inner_opt_builder.overrides,
        ) as (fnet, diffopt):
            # Inner-loop updates on the support set
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt)
                spt_loss = F.cross_entropy(spt_pred, y_spt)
                diffopt.step(spt_loss)

            # Query set evaluation
            qry_pred = fnet(x_qry)
            qry_loss = F.cross_entropy(qry_pred, y_qry)
            qry_losses.append(qry_loss.detach().cpu().numpy())

            # Comptage des erreurs par angle
            qry_pred_classes = torch.argmax(qry_pred, dim=1)  # Prédictions des classes
            for pred, label, angle in zip(qry_pred_classes, y_qry, angles_qry):
                angle_val = angle.item()
                angle_counts[angle_val] += 1  # Incrémenter compteur d'images
                if pred != label:  # Si prédiction incorrecte
                    angle_errors[angle_val] += 1

    avg_qry_loss = np.mean(qry_losses)
    _low, high = st.t.interval(
        0.95, len(qry_losses) - 1, loc=avg_qry_loss, scale=st.sem(qry_losses)
    )
    test_metrics = {"test_loss": avg_qry_loss, "test_err": high - avg_qry_loss}
    wandb.log(test_metrics, step=step_idx)

   # Log des erreurs et du nombre d'images par angle
    for angle in angle_counts.keys():
        wandb.log({f"errors_angle_{angle}": angle_errors[angle]}, step=step_idx)
    
    # Générer le graphique en barres pour le nombre d'images par angle
    angles = list(angle_counts.keys())
    counts = list(angle_counts.values())
    plt.figure(figsize=(10, 6))
    plt.bar(angles, counts, color='skyblue')
    plt.xlabel("Angle de Rotation (°)")
    plt.ylabel("Nombre d'Images")
    plt.title("Nombre d'Images par Angle de Rotation (TEST)")
    plt.xticks(angles)
    plt.tight_layout()
    # Log du graphique directement dans wandb
    wandb.log({"angle_counts_barplot2": wandb.Image(plt)}, step=step_idx)
    plt.close()

    # /!\ Pour visualiser le weight sharing de test, ne pas mettre en même temps que le train.
    # **Appel à visualize_weights et visualize_activations**
    if weight and (step_idx == 0 or (step_idx + 1) % 100 == 0):
        # Visualiser les poids appris
        visualize_weights(net, architecture, step_idx)

        # Visualiser les activations intermédiaires pour un échantillon (query set)
        x_sample = task_data[0][3][:1]  # Prendre un échantillon du query set
        visualize_activations(net, x_sample, architecture, step_idx)

        
    return avg_qry_loss

def build_model(architecture, device, num_tasks):
    """Construire le modèle selon l'architecture spécifiée."""
    if architecture == "shared":
        # Toutes les rotations aplaties en entrée d'un unique circuit
        net = torch.nn.Sequential(
            layers.ShareConv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            layers.ShareLinearFull(32 * 26 * 26, 10)  
            # 10 classes pour RMNIST, 32 filtres, 26=28 (taille image)-3 (taille filtre)+1
        ).to(device)
    elif architecture == "dedicated":
        # Chaque angle de rotation avec un circuit dédié (avec une couche d'alignement en sortie)
        circuits = nn.ModuleList()
        for _ in range(num_tasks):
            circuit = nn.Sequential(
                layers.ShareConv2d(1, 32, kernel_size=3),
                nn.ReLU(),
                nn.Flatten()
            )
            circuits.append(circuit)

        # Couche d'alignement en sortie
        align_layer = layers.ShareLinearFull(32 * 26 * 26 * num_tasks, 10)

        class DedicatedNetwork(nn.Module):
            def __init__(self, circuits, align_layer):
                super(DedicatedNetwork, self).__init__()
                self.circuits = circuits
                self.align_layer = align_layer

            def forward(self, x):
                # Appliquer chaque circuit dédié sur l'entrée
                outputs = []
                for circuit in self.circuits:
                    outputs.append(circuit(x))
                # Concatenation des sorties des circuits
                concatenated = torch.cat(outputs, dim=-1)
                # Passer par la couche d'alignement
                return self.align_layer(concatenated)

        net = DedicatedNetwork(circuits, align_layer).to(device)
    else:
        raise ValueError("Architecture inconnue : " + architecture)
    return net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_inner_lr", type=float, default=0.01)
    parser.add_argument("--outer_lr", type=float, default=0.0001)
    parser.add_argument("--k_spt", type=int, default=1)
    parser.add_argument("--k_qry", type=int, default=19)
    parser.add_argument("--lr_mode", type=str, default="per_layer")
    parser.add_argument("--num_inner_steps", type=int, default=1)
    parser.add_argument("--num_outer_steps", type=int, default=1000)
    parser.add_argument("--inner_opt", type=str, default="maml")
    parser.add_argument("--outer_opt", type=str, default="Adam")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_tasks", type=int, default=5) #Nombre d'angles différents
    parser.add_argument("--per_task_rotation", type=int, default=9 ) #Décalage de l'angle
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--architecture", type=str, choices=["shared", "dedicated"], default="shared")
    parser.add_argument("--view_weight_sharing", type=bool, default=False) #voir weigt sharing ou non

    args = parser.parse_args()

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    wandb.init(project="rmnist_meta_learning", dir=OUTPUT_PATH)
    wandb.config.update(args)
    cfg = wandb.config

    device = torch.device(cfg.device)

    #Création données RMNIST
    train_loader, test_loader = flattened_rotMNIST(
        num_tasks=cfg.num_tasks,
        per_task_rotation=cfg.per_task_rotation,
        batch_size=cfg.batch_size
    )

    #Transformation des données pour adaptation
    train_task_data = load_rmnist_task_data(train_loader, cfg.num_tasks, cfg.k_spt, cfg.k_qry)
    test_task_data = load_rmnist_task_data(test_loader, cfg.num_tasks, cfg.k_spt, cfg.k_qry)

    # Define model
    net = build_model(cfg.architecture, device, cfg.num_tasks)

    inner_opt_builder = InnerOptBuilder(
        net, device, cfg.inner_opt, cfg.init_inner_lr, "learned", cfg.lr_mode
    )

    if cfg.outer_opt == "SGD":
        meta_opt = optim.SGD(inner_opt_builder.metaparams.values(), lr=cfg.outer_lr)
    else:
        meta_opt = optim.Adam(inner_opt_builder.metaparams.values(), lr=cfg.outer_lr)

    start_time = time.time()
    for step_idx in range(cfg.num_outer_steps):
        train(
            step_idx,
            train_task_data,
            net,
            inner_opt_builder,
            meta_opt,
            cfg.num_inner_steps,
            cfg.architecture,
            cfg.view_weight_sharing
        )

        if step_idx == 0 or (step_idx + 1) % 100 == 0:
            val_loss = test(
                step_idx,
                test_task_data,
                net,
                inner_opt_builder,
                cfg.num_inner_steps,
                cfg.architecture,
                cfg.view_weight_sharing
            )

            steps_p_sec = (step_idx + 1) / (time.time() - start_time)
            wandb.log({"steps_per_sec": steps_p_sec}, step=step_idx)
            print(f"Step: {step_idx}. Steps/sec: {steps_p_sec:.2f}. Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()
