import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE # For t-SNE visualization
from scipy.spatial import ConvexHull # For drawing cluster boundaries
import copy
import random
import matplotlib.pyplot as plt
import matplotlib as mpl # For more control over rcParams

# --- Apply a style and set some global parameters for美化 ---
# plt.style.use('seaborn-v0_8-whitegrid') 
# plt.style.use('ggplot')
plt.style.use('bmh') 
mpl.rcParams['font.family'] = 'sans-serif' 
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 6


# --- Configuration ---
NUM_CLIENTS = 20
NUM_CLUSTERS = 3
NUM_FEATURES_GLOBAL = 5
NUM_FEATURES_CLUSTER = 5
NUM_FEATURES_PERSONAL = 5

SAMPLES_PER_CLIENT = 100 # Or your increased value like 500
COMMUNICATION_ROUNDS = 50
LOCAL_EPOCHS = 3
CLIENT_LEARNING_RATE = 0.01
SERVER_LEARNING_RATE_GB = 1.0
SERVER_LEARNING_RATE_CS = 1.0
DYNAMIC_CLUSTERING_INTERVAL = 10
SEED = 42

USE_NON_LINEAR_MODELS = True
USE_NON_LINEAR_DATA = True
MLP_HIDDEN_DIM = 16

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- 1. Synthetic Data Generation ---
def non_linear_transform_data(X_raw, W_true):
    linear_combination = X_raw @ W_true
    return np.sin(linear_combination)

def generate_synthetic_data(num_clients, samples_per_client, num_clusters,
                            n_fg, n_fc, n_fp, use_non_linear_data_gen=USE_NON_LINEAR_DATA):
    clients_data = []
    clients_true_clusters = []
    true_w_global = np.random.randn(n_fg, 1) * 0.5
    true_w_clusters = [np.random.randn(n_fc, 1) * 0.5 for _ in range(num_clusters)]
    true_w_personals = [np.random.randn(n_fp, 1) * 0.5 for _ in range(num_clients)]

    for i in range(num_clients):
        true_cluster_idx = i % num_clusters
        clients_true_clusters.append(true_cluster_idx)
        X_global_raw = np.random.rand(samples_per_client, n_fg) * 2 - 1
        X_cluster_raw = np.random.rand(samples_per_client, n_fc) * 2 - 1
        X_personal_raw = np.random.rand(samples_per_client, n_fp) * 2 - 1
        
        if use_non_linear_data_gen:
            y_global_effect = non_linear_transform_data(X_global_raw, true_w_global)
            y_cluster_effect = non_linear_transform_data(X_cluster_raw, true_w_clusters[true_cluster_idx])
            y_personal_effect = non_linear_transform_data(X_personal_raw, true_w_personals[i])
        else:
            y_global_effect = X_global_raw @ true_w_global
            y_cluster_effect = X_cluster_raw @ true_w_clusters[true_cluster_idx]
            y_personal_effect = X_personal_raw @ true_w_personals[i]
        
        noise_y = 0.1
        y = y_global_effect + y_cluster_effect + y_personal_effect + np.random.randn(samples_per_client, 1) * noise_y
        X_combined_raw = np.concatenate((X_global_raw, X_cluster_raw, X_personal_raw), axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined_raw)
        clients_data.append({
            'X': torch.tensor(X_scaled, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'true_cluster': true_cluster_idx, 'scaler': scaler})
    return clients_data, clients_true_clusters

# --- 2. Hierarchical Model Structure ---
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x): return self.linear(x)

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=MLP_HIDDEN_DIM, output_dim=1):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def get_model_component(input_dim, use_mlp=USE_NON_LINEAR_MODELS):
    return MLPModel(input_dim) if use_mlp else LinearModel(input_dim)

class HierarchicalClientModel(nn.Module):
    def __init__(self, gb_model, cs_model, ph_model, n_fg_idx_end, n_fc_idx_end):
        super(HierarchicalClientModel, self).__init__()
        self.gb_model, self.cs_model, self.ph_model = gb_model, cs_model, ph_model
        self.n_fg_idx_end, self.n_fc_idx_end = n_fg_idx_end, n_fc_idx_end
    def forward(self, x_combined):
        x_gb,x_cs,x_ph = (x_combined[:,:self.n_fg_idx_end], 
                         x_combined[:,self.n_fg_idx_end:self.n_fc_idx_end], 
                         x_combined[:,self.n_fc_idx_end:])
        return self.gb_model(x_gb) + self.cs_model(x_cs) + self.ph_model(x_ph)

# --- Client Class ---
class Client:
    def __init__(self, client_id, local_data, n_fg, n_fc, n_fp, client_lr):
        self.client_id, self.local_data, self.lr = client_id, local_data, client_lr
        self.n_fg, self.n_fc, self.n_fp = n_fg, n_fc, n_fp
        self.n_fg_idx_end, self.n_fc_idx_end = n_fg, n_fg + n_fc
        self.ph_model = get_model_component(self.n_fp)
        self.optimizer_ph_only = optim.SGD(self.ph_model.parameters(), lr=self.lr)
        self.gb_model_local_copy, self.cs_model_local_copy, self.cluster_id = None, None, -1

    def set_models(self, gb_model_state_dict, cs_model_state_dict, cluster_id):
        self.gb_model_local_copy = get_model_component(self.n_fg)
        self.gb_model_local_copy.load_state_dict(gb_model_state_dict)
        self.cs_model_local_copy = get_model_component(self.n_fc)
        self.cs_model_local_copy.load_state_dict(cs_model_state_dict)
        self.cluster_id = cluster_id

    def local_train(self, local_epochs):
        if not self.gb_model_local_copy or not self.cs_model_local_copy: return None, None, 0
        hier_model = HierarchicalClientModel(self.gb_model_local_copy, self.cs_model_local_copy, self.ph_model,
                                           self.n_fg_idx_end, self.n_fc_idx_end)
        optimizer = optim.SGD(hier_model.parameters(), lr=self.lr); criterion = nn.MSELoss()
        initial_gb_state = copy.deepcopy(self.gb_model_local_copy.state_dict())
        initial_cs_state = copy.deepcopy(self.cs_model_local_copy.state_dict())
        hier_model.train()
        for _ in range(local_epochs):
            X_b, y_b = self.local_data['X'], self.local_data['y']
            optimizer.zero_grad(); preds = hier_model(X_b); loss = criterion(preds, y_b)
            loss.backward(); optimizer.step()
        gb_delta = {n: self.gb_model_local_copy.state_dict()[n]-initial_gb_state[n] for n in initial_gb_state}
        cs_delta = {n: self.cs_model_local_copy.state_dict()[n]-initial_cs_state[n] for n in initial_cs_state}
        return gb_delta, cs_delta, len(self.local_data['X'])

    def local_train_ph_only(self, local_epochs, gb_ref, cs_ref):
        criterion = nn.MSELoss(); self.ph_model.train(); gb_ref.eval(); cs_ref.eval()
        for _ in range(local_epochs):
            X_comb, y_b = self.local_data['X'], self.local_data['y']
            self.optimizer_ph_only.zero_grad()
            x_gb,x_cs,x_ph = (X_comb[:,:self.n_fg_idx_end],X_comb[:,self.n_fg_idx_end:self.n_fc_idx_end],X_comb[:,self.n_fc_idx_end:])
            with torch.no_grad(): out_gb, out_cs = gb_ref(x_gb), cs_ref(x_cs)
            out_ph = self.ph_model(x_ph); preds = out_gb + out_cs + out_ph
            loss = criterion(preds, y_b); loss.backward(); self.optimizer_ph_only.step()
        return len(self.local_data['X'])

    def get_ph_params_flat(self):
        params = [p.data.view(-1) for p in self.ph_model.parameters()]
        return torch.cat(params) if params else torch.empty(0)

    def evaluate(self, gb_model_server, cs_model_for_cluster_server):
        hier_model_eval = HierarchicalClientModel(gb_model_server, cs_model_for_cluster_server, self.ph_model,
                                               self.n_fg_idx_end, self.n_fc_idx_end)
        hier_model_eval.eval(); criterion = nn.MSELoss()
        with torch.no_grad(): preds = hier_model_eval(self.local_data['X']); loss = criterion(preds, self.local_data['y'])
        return loss.item() * len(self.local_data['X']), len(self.local_data['X'])

# --- Server Class ---
class Server:
    def __init__(self, num_clients, num_clusters, n_fg, n_fc, n_fp, client_data_list, client_lr, initial_cluster_assignments=None):
        self.num_clients, self.num_clusters = num_clients, num_clusters
        self.n_fg, self.n_fc, self.n_fp = n_fg, n_fc, n_fp
        self.gb_model = get_model_component(n_fg)
        self.cs_models = [get_model_component(n_fc) for _ in range(num_clusters)]
        self.clients = [Client(i, client_data_list[i], n_fg, n_fc, n_fp, client_lr) for i in range(num_clients)]
        self.cluster_assignments = list(initial_cluster_assignments) if initial_cluster_assignments \
                                   else [random.randint(0,num_clusters-1) for _ in range(num_clients)]
        for client_idx, cluster_idx in enumerate(self.cluster_assignments):
            self.clients[client_idx].cluster_id = cluster_idx
        print(f"Initial cluster assignments: {self.cluster_assignments}")
        self.history_per_cluster_losses = {k: [] for k in range(num_clusters)}
        self.history_cluster_loss_rounds = []


    def dynamic_cluster(self):
        print("Performing dynamic clustering...")
        if self.num_clusters <= 1: return
        ph_params = [c.get_ph_params_flat().cpu().detach().numpy() for c in self.clients]
        ph_params_f = [p for p in ph_params if p.size>0]; client_idx_f = [i for i,p in enumerate(ph_params) if p.size>0]
        if not ph_params_f: print("No valid PH params."); return
        ph_matrix = np.array(ph_params_f)
        eff_k = max(1,ph_matrix.shape[0]) if ph_matrix.shape[0]<self.num_clusters else self.num_clusters
        try:
            kmeans = KMeans(n_clusters=eff_k, random_state=SEED, n_init='auto')
            if client_idx_f:
                new_labels_subset = kmeans.fit_predict(ph_matrix)
                new_assign_full = list(self.cluster_assignments)
                for i, orig_idx in enumerate(client_idx_f): new_assign_full[orig_idx] = new_labels_subset[i]
                self.cluster_assignments = new_assign_full
                for c_idx, cl_idx in enumerate(self.cluster_assignments): self.clients[c_idx].cluster_id = cl_idx
                print(f"New cluster assignments: {self.cluster_assignments}")
        except Exception as e: print(f"KMeans Error: {e}. Keeping old.")

    def evaluate_clients_and_log_per_cluster(self, current_round):
        total_loss, total_samples = 0, 0
        current_losses_by_cluster = {k: [] for k in range(self.num_clusters)}
        client_losses_this_round = []

        for client_idx, client in enumerate(self.clients):
            client_cluster_id = self.cluster_assignments[client_idx]
            loss_val_weighted, num_s = client.evaluate(self.gb_model, self.cs_models[client_cluster_id])
            client_loss = loss_val_weighted / num_s if num_s > 0 else 0
            client_losses_this_round.append(client_loss)
            current_losses_by_cluster[client_cluster_id].append(client_loss)
            total_loss += loss_val_weighted; total_samples += num_s
        
        avg_loss_overall = total_loss / total_samples if total_samples > 0 else float('inf')
        print(f"Average evaluation loss across all clients: {avg_loss_overall:.4f}")

        if (current_round + 1) % 1 == 0: 
            self.history_cluster_loss_rounds.append(current_round + 1)
            for k_cluster in range(self.num_clusters):
                if current_losses_by_cluster[k_cluster]:
                    avg_loss_k = np.mean(current_losses_by_cluster[k_cluster])
                    self.history_per_cluster_losses[k_cluster].append(avg_loss_k)
                else:
                    self.history_per_cluster_losses[k_cluster].append(np.nan)
        return avg_loss_overall


    def train_round(self, current_round, local_epochs, dynamic_clustering_interval):
        print(f"\n--- Communication Round {current_round + 1} ---")
        if dynamic_clustering_interval and (current_round+1)%dynamic_clustering_interval==0 and current_round>0:
            self.dynamic_cluster()
        gb_deltas, cs_deltas_clustered = [], [[] for _ in range(self.num_clusters)]
        gb_counts, cs_counts_clustered = [], [[] for _ in range(self.num_clusters)]
        for c_idx, client in enumerate(self.clients):
            cl_id = self.cluster_assignments[c_idx]
            client.set_models(copy.deepcopy(self.gb_model.state_dict()), 
                              copy.deepcopy(self.cs_models[cl_id].state_dict()), cl_id)
            gb_d, cs_d, n_s = client.local_train(local_epochs)
            if gb_d and cs_d:
                gb_deltas.append(gb_d); gb_counts.append(n_s)
                cs_deltas_clustered[cl_id].append(cs_d); cs_counts_clustered[cl_id].append(n_s)
        if gb_deltas:
            total_s_gb = sum(gb_counts)
            avg_gb_d = {n:torch.zeros_like(self.gb_model.state_dict()[n]) for n in self.gb_model.state_dict()}
            for i,d in enumerate(gb_deltas):
                w = gb_counts[i]/total_s_gb if total_s_gb>0 else 1/len(gb_deltas)
                for n in avg_gb_d: avg_gb_d[n] += d[n]*w
            upd_gb_s = {n:self.gb_model.state_dict()[n]+SERVER_LEARNING_RATE_GB*avg_gb_d[n] for n in self.gb_model.state_dict()}
            self.gb_model.load_state_dict(upd_gb_s)
        for k in range(self.num_clusters):
            if cs_deltas_clustered[k]:
                total_s_cs_k = sum(cs_counts_clustered[k])
                avg_cs_d_k = {n:torch.zeros_like(self.cs_models[k].state_dict()[n]) for n in self.cs_models[k].state_dict()}
                for i,d in enumerate(cs_deltas_clustered[k]):
                    w = cs_counts_clustered[k][i]/total_s_cs_k if total_s_cs_k>0 else 1/len(cs_deltas_clustered[k])
                    for n in avg_cs_d_k: avg_cs_d_k[n] += d[n]*w
                upd_cs_s_k = {n:self.cs_models[k].state_dict()[n]+SERVER_LEARNING_RATE_CS*avg_cs_d_k[n] for n in self.cs_models[k].state_dict()}
                self.cs_models[k].load_state_dict(upd_cs_s_k)
        return self.evaluate_clients_and_log_per_cluster(current_round)

# --- Main Execution ---
print("Generating synthetic data...")
client_datasets, true_client_clusters = generate_synthetic_data(
    NUM_CLIENTS, SAMPLES_PER_CLIENT, NUM_CLUSTERS,
    NUM_FEATURES_GLOBAL, NUM_FEATURES_CLUSTER, NUM_FEATURES_PERSONAL)
print(f"True client cluster distribution: {true_client_clusters}")
print(f"Using non-linear models: {USE_NON_LINEAR_MODELS}, Using non-linear data generation: {USE_NON_LINEAR_DATA}")

print("\n=== Running Fed-ITS-Hybrid ===")
server_hybrid = Server(NUM_CLIENTS, NUM_CLUSTERS, NUM_FEATURES_GLOBAL, NUM_FEATURES_CLUSTER, 
                       NUM_FEATURES_PERSONAL, client_datasets, CLIENT_LEARNING_RATE)
fed_its_hybrid_losses = []
ari_scores_over_rounds_hybrid = [] 
for r in range(COMMUNICATION_ROUNDS):
    loss = server_hybrid.train_round(r, LOCAL_EPOCHS, DYNAMIC_CLUSTERING_INTERVAL)
    fed_its_hybrid_losses.append(loss)
    if DYNAMIC_CLUSTERING_INTERVAL and (r + 1) % DYNAMIC_CLUSTERING_INTERVAL == 0 and r >= 0: 
        if len(set(server_hybrid.cluster_assignments)) > 1 and len(set(true_client_clusters)) > 1:
            ari = adjusted_rand_score(true_client_clusters, server_hybrid.cluster_assignments)
            print(f"Round {r+1} ARI (vs true clusters): {ari:.3f}")
            ari_scores_over_rounds_hybrid.append({'round': r + 1, 'ari': ari})

print("\n=== Running Standard FedAvg ===")
TOTAL_FEATURES_COMBINED = NUM_FEATURES_GLOBAL + NUM_FEATURES_CLUSTER + NUM_FEATURES_PERSONAL
fedavg_global_model = get_model_component(TOTAL_FEATURES_COMBINED, use_mlp=USE_NON_LINEAR_MODELS)
fedavg_losses = []
for r in range(COMMUNICATION_ROUNDS):
    print(f"\n--- FedAvg Round {r + 1} ---")
    client_deltas_f, client_samples_f = [], []; current_loss_f, total_samples_f = 0,0
    for client_idx in range(NUM_CLIENTS):
        client_data = client_datasets[client_idx]
        local_m_f = copy.deepcopy(fedavg_global_model); opt_f = optim.SGD(local_m_f.parameters(),lr=CLIENT_LEARNING_RATE)
        crit_f = nn.MSELoss(); initial_s_f = copy.deepcopy(local_m_f.state_dict()); local_m_f.train()
        for _ in range(LOCAL_EPOCHS):
            opt_f.zero_grad(); pred_f = local_m_f(client_data['X']); loss_f = crit_f(pred_f, client_data['y'])
            loss_f.backward(); opt_f.step()
        delta_f = {n:local_m_f.state_dict()[n]-initial_s_f[n] for n in initial_s_f}
        client_deltas_f.append(delta_f); client_samples_f.append(len(client_data['X']))
        local_m_f.eval(); 
        with torch.no_grad(): preds_eval_f = local_m_f(client_data['X'])
        loss_val_f = crit_f(preds_eval_f, client_data['y'])
        current_loss_f += loss_val_f.item()*len(client_data['X']); total_samples_f += len(client_data['X'])
    if client_deltas_f:
        total_s = sum(client_samples_f)
        avg_d = {n:torch.zeros_like(fedavg_global_model.state_dict()[n]) for n in fedavg_global_model.state_dict()}
        for i,d in enumerate(client_deltas_f):
            w = client_samples_f[i]/total_s if total_s > 0 else 1/len(client_deltas_f)
            for n in avg_d: avg_d[n] += d[n]*w
        upd_s = {n:fedavg_global_model.state_dict()[n]+SERVER_LEARNING_RATE_GB*avg_d[n] for n in fedavg_global_model.state_dict()}
        fedavg_global_model.load_state_dict(upd_s)
    fedavg_losses.append(current_loss_f/total_samples_f if total_samples_f>0 else float('inf'))
    print(f"FedAvg Round {r+1} Avg Loss: {fedavg_losses[-1]:.4f}")

print("\n=== Running Local Training Only (Structured PH) ===")
local_clients_ph_only = [Client(i, client_datasets[i], NUM_FEATURES_GLOBAL, NUM_FEATURES_CLUSTER, 
                                NUM_FEATURES_PERSONAL, CLIENT_LEARNING_RATE) for i in range(NUM_CLIENTS)]
local_only_losses = []
fixed_gb_dummy = get_model_component(NUM_FEATURES_GLOBAL, use_mlp=USE_NON_LINEAR_MODELS)
for p in fixed_gb_dummy.parameters(): p.data.fill_(0); p.requires_grad=False
fixed_gb_dummy.eval()
fixed_cs_dummy = get_model_component(NUM_FEATURES_CLUSTER, use_mlp=USE_NON_LINEAR_MODELS)
for p in fixed_cs_dummy.parameters(): p.data.fill_(0); p.requires_grad=False
fixed_cs_dummy.eval()
for r in range(COMMUNICATION_ROUNDS):
    print(f"\n--- LocalOnly Round {r+1} ---")
    current_loss_l, total_samples_l = 0,0
    for client in local_clients_ph_only:
        client.local_train_ph_only(LOCAL_EPOCHS, fixed_gb_dummy, fixed_cs_dummy)
        loss_l, num_s_l = client.evaluate(fixed_gb_dummy, fixed_cs_dummy)
        current_loss_l += loss_l; total_samples_l += num_s_l
    local_only_losses.append(current_loss_l/total_samples_l if total_samples_l > 0 else float('inf'))
    print(f"LocalOnly Round {r+1} Avg Loss: {local_only_losses[-1]:.4f}")

# --- Visualization Section ---
print("\n--- Generating Visualizations ---")

plt.figure(figsize=(9, 6))
plt.plot(fed_its_hybrid_losses, label='Fed-ITS-Hybrid', marker='o', linestyle='-')
plt.plot(fedavg_losses, label='FedAvg', marker='s', linestyle='--')
plt.plot(local_only_losses, label='Local Only (Structured PH)', marker='^', linestyle=':')
plt.xlabel('Communication Round'); plt.ylabel('Average MSE Loss')
plt.title(f'FL Performance (Data: {"Non-linear" if USE_NON_LINEAR_DATA else "Linear"}, Models: {"MLP" if USE_NON_LINEAR_MODELS else "Linear"})')
plt.legend(frameon=True, loc='upper right'); plt.grid(True, linestyle='--', alpha=0.7); plt.yscale('log'); plt.tight_layout()
plt.savefig("fl_performance_curves.pdf", format='pdf', bbox_inches='tight'); print("Saved: fl_performance_curves.pdf"); plt.show()

if ari_scores_over_rounds_hybrid:
    rounds_ari = [item['round'] for item in ari_scores_over_rounds_hybrid]
    aris = [item['ari'] for item in ari_scores_over_rounds_hybrid]
    plt.figure(figsize=(8, 5))
    plt.plot(rounds_ari, aris, marker='o', linestyle='-', color='teal')
    plt.title('Fed-ITS-Hybrid: ARI Over Communication Rounds'); plt.xlabel('Communication Round of Clustering Event')
    plt.ylabel('ARI Score'); plt.ylim(min(0, min(aris)-0.05) if aris else 0, max(1, max(aris)+0.05) if aris else 1)
    plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig("fed_its_hybrid_ari_over_rounds.pdf", format='pdf', bbox_inches='tight')
    print("Saved: fed_its_hybrid_ari_over_rounds.pdf"); plt.show()

def visualize_ph_params_tsne(server, true_clusters, title="PH Params t-SNE", filename_base="tsne_ph_params"):
    if not server.clients: print("No clients for t-SNE."); return
    all_ph_params, client_ids, current_assigns_list, true_cls_plot = [],[],[],[]
    for idx, client in enumerate(server.clients):
        ph_params_flat = client.get_ph_params_flat().cpu().detach().numpy()
        if ph_params_flat.size > 0:
            all_ph_params.append(ph_params_flat); client_ids.append(client.client_id)
            current_assigns_list.append(server.cluster_assignments[idx])
            true_cls_plot.append(true_clusters[idx])
    if not all_ph_params: print("No PH params for t-SNE."); return
    all_ph_matrix = np.array(all_ph_params)
    current_assigns_array = np.array(current_assigns_list)
    if all_ph_matrix.shape[0] < 2: print("Not enough samples for t-SNE."); return
    print(f"Running t-SNE on {all_ph_matrix.shape[0]} PH params...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30.0, all_ph_matrix.shape[0]-1.0), 
                max_iter=300, learning_rate='auto', init='pca') # max_iter used
    ph_2d = tsne.fit_transform(all_ph_matrix)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Corrected colormap retrieval using new API with fallback
    try:
        cmap_learned_base = mpl.colormaps.get_cmap('viridis')
        cmap_true_base = mpl.colormaps.get_cmap('plasma')
    except AttributeError: # Fallback for older Matplotlib
        cmap_learned_base = plt.cm.get_cmap('viridis')
        cmap_true_base = plt.cm.get_cmap('plasma')

    unique_learned_clusters = np.unique(current_assigns_array)
    legend_handles_learned = []
    axes[0].set_title(f'{title} - Learned Clusters (K={server.num_clusters})')
    axes[0].set_xlabel("t-SNE Dimension 1"); axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    learned_label_to_idx = {label: i for i, label in enumerate(unique_learned_clusters)}

    for cluster_label in unique_learned_clusters: # Iterate over unique labels
        cluster_points = ph_2d[current_assigns_array == cluster_label]
        # Use the mapped index for consistent color from the base colormap
        color_norm = learned_label_to_idx[cluster_label] / (len(unique_learned_clusters) - 1 if len(unique_learned_clusters) > 1 else 1)
        color = cmap_learned_base(color_norm if len(unique_learned_clusters) > 1 else 0.5) # Handle single cluster case for norm
        
        handle = axes[0].scatter(cluster_points[:,0], cluster_points[:,1], color=color, 
                                 alpha=0.7, s=50, label=f'Learned {cluster_label}')
        legend_handles_learned.append(handle)
        if cluster_points.shape[0] >= 3:
            try:
                hull = ConvexHull(cluster_points)
                for simplex in hull.simplices:
                    axes[0].plot(cluster_points[simplex,0], cluster_points[simplex,1], '-', color=color, lw=1.5, alpha=0.8)
            except Exception as e: print(f"CHull Error (Learned Cl {cluster_label}): {e}")
    axes[0].legend(handles=legend_handles_learned, title="Learned Clusters", frameon=True, loc='best')

    unique_true_clusters = np.unique(true_cls_plot)
    legend_handles_true = []
    axes[1].set_title(f'{title} - True Clusters (K_true={len(set(true_clusters))})')
    axes[1].set_xlabel("t-SNE Dimension 1"); axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    true_label_to_idx = {label: i for i, label in enumerate(unique_true_clusters)}

    for cluster_label in unique_true_clusters: # Iterate over unique labels
        cluster_points = ph_2d[np.array(true_cls_plot) == cluster_label]
        color_norm = true_label_to_idx[cluster_label] / (len(unique_true_clusters) - 1 if len(unique_true_clusters) > 1 else 1)
        color = cmap_true_base(color_norm if len(unique_true_clusters) > 1 else 0.5) # Handle single cluster case for norm
        
        handle = axes[1].scatter(cluster_points[:,0],cluster_points[:,1], color=color, alpha=0.7, s=50, label=f'True {cluster_label}')
        legend_handles_true.append(handle)
    axes[1].legend(handles=legend_handles_true, title="True Clusters", frameon=True, loc='best')
    plt.tight_layout(); plt.savefig(f"{filename_base}.pdf", format='pdf', bbox_inches='tight')
    print(f"Saved: {filename_base}.pdf"); plt.show()

if fed_its_hybrid_losses: 
    visualize_ph_params_tsne(server_hybrid, true_client_clusters, 
                             title=f"Fed-ITS-Hybrid PH Params (End, R{COMMUNICATION_ROUNDS})",
                             filename_base="fed_its_hybrid_tsne_end")

if fed_its_hybrid_losses and server_hybrid.history_cluster_loss_rounds:
    plt.figure(figsize=(10, 6))
    
    # Corrected colormap retrieval for per-cluster loss
    cluster_colors_list = []
    try:
        if NUM_CLUSTERS <= 10:
            base_cmap_per_cluster = mpl.colormaps.get_cmap('tab10')
            cluster_colors_list = [base_cmap_per_cluster(i) for i in range(NUM_CLUSTERS)]
        else: 
            base_cmap_per_cluster = mpl.colormaps.get_cmap('viridis')
            cluster_colors_list = [base_cmap_per_cluster(i / (NUM_CLUSTERS -1 if NUM_CLUSTERS > 1 else 1)) for i in range(NUM_CLUSTERS)]
    except AttributeError: # Fallback for older Matplotlib
        if NUM_CLUSTERS <= 10:
            base_cmap_per_cluster = plt.cm.get_cmap('tab10')
            cluster_colors_list = [base_cmap_per_cluster(i) for i in range(NUM_CLUSTERS)]
        else:
            base_cmap_per_cluster = plt.cm.get_cmap('viridis')
            cluster_colors_list = [base_cmap_per_cluster(i / (NUM_CLUSTERS -1 if NUM_CLUSTERS > 1 else 1)) for i in range(NUM_CLUSTERS)]


    for k_cluster in range(NUM_CLUSTERS): # k_cluster will be 0, 1, 2 for NUM_CLUSTERS=3
        valid_rounds = server_hybrid.history_cluster_loss_rounds
        valid_losses_k = [l for l in server_hybrid.history_per_cluster_losses[k_cluster] if not np.isnan(l)]
        corresponding_rounds_k = [r for r, l in zip(valid_rounds, server_hybrid.history_per_cluster_losses[k_cluster]) if not np.isnan(l)]
        if corresponding_rounds_k: # Only plot if there's data for this cluster
             plt.plot(corresponding_rounds_k, valid_losses_k, marker='.', linestyle='-', 
                      label=f'Cluster {k_cluster} Avg Loss', color=cluster_colors_list[k_cluster % len(cluster_colors_list)]) 
    
    plt.title('Fed-ITS-Hybrid: Average Loss per Cluster Over Rounds')
    plt.xlabel('Communication Round'); plt.ylabel('Average MSE Loss')
    plt.legend(frameon=True, loc='best'); plt.grid(True, linestyle='--', alpha=0.7); plt.yscale('log'); plt.tight_layout()
    plt.savefig("fed_its_hybrid_per_cluster_loss.pdf", format='pdf', bbox_inches='tight')
    print("Saved: fed_its_hybrid_per_cluster_loss.pdf"); plt.show()

def visualize_component_contributions(server, client_id_to_viz, sample_idx_to_viz, title="Component Contributions", filename_base="component_contributions"):
    if client_id_to_viz >= len(server.clients): print(f"Client ID {client_id_to_viz} out of range."); return
    client_obj = server.clients[client_id_to_viz]; client_cluster_id = server.cluster_assignments[client_id_to_viz]
    if sample_idx_to_viz >= len(client_obj.local_data['X']): print(f"Sample ID {sample_idx_to_viz} out of range."); return
    X_sample_combined = client_obj.local_data['X'][sample_idx_to_viz].unsqueeze(0)
    y_true_sample = client_obj.local_data['y'][sample_idx_to_viz].item()
    gb_model,cs_model,ph_model = server.gb_model, server.cs_models[client_cluster_id], client_obj.ph_model
    gb_model.eval(); cs_model.eval(); ph_model.eval()
    with torch.no_grad():
        x_gb,x_cs,x_ph = (X_sample_combined[:,:client_obj.n_fg_idx_end],X_sample_combined[:,client_obj.n_fg_idx_end:client_obj.n_fc_idx_end],X_sample_combined[:,client_obj.n_fc_idx_end:])
        out_gb,out_cs,out_ph = gb_model(x_gb).item(),cs_model(x_cs).item(),ph_model(x_ph).item()
        y_pred_total = out_gb + out_cs + out_ph
    labels,contributions = ['GB','CS','PH'],[out_gb,out_cs,out_ph]
    component_colors = ['cornflowerblue','mediumseagreen','lightcoral']
    fig,ax = plt.subplots(figsize=(8,5.5))
    bars = ax.bar(labels,contributions,color=component_colors,edgecolor='grey')
    ax.axhline(0,color='black',lw=0.8); ax.axhline(y_true_sample,color='firebrick',ls='--',lw=1.5,label=f'True y ({y_true_sample:.2f})')
    ax.axhline(y_pred_total,color='darkblue',ls=':',lw=1.5,label=f'Total Pred ({y_pred_total:.2f})')
    ax.set_ylabel('Output Value'); ax.set_title(f'{title}\nC{client_id_to_viz},S{sample_idx_to_viz},Clus{client_cluster_id}'); ax.legend(frameon=True,loc='best')
    ax.grid(axis='y',ls='--',alpha=0.7)
    for bar in bars: yval=bar.get_height(); plt.text(bar.get_x()+bar.get_width()/2.0,yval+(0.02 if yval>=0 else -0.06)*(max(abs(yval),0.1)+abs(y_true_sample)),f'{yval:.2f}',ha='center',va='bottom' if yval>=0 else 'top',fontsize=9)
    plt.tight_layout(); plt.savefig(f"{filename_base}_c{client_id_to_viz}_s{sample_idx_to_viz}.pdf",format='pdf',bbox_inches='tight')
    print(f"Saved: {filename_base}_c{client_id_to_viz}_s{sample_idx_to_viz}.pdf"); plt.show()

# --- Example call for component contributions (uncomment to use) ---
# if fed_its_hybrid_losses:
#     CLIENT_TO_VISUALIZE = 0; SAMPLE_TO_VISUALIZE = 0
#     visualize_component_contributions(server_hybrid, CLIENT_TO_VISUALIZE, SAMPLE_TO_VISUALIZE,
#                                       title="Fed-ITS-Hybrid Components (End of Training)",
#                                       filename_base="components_contributions")

def visualize_1d_ph_function(client_obj, title="1D PH Model Function", filename_base="1d_ph_function"):
    if client_obj.n_fp != 1: print("1D PH viz for 1D input PH only."); return
    ph_model = client_obj.ph_model; ph_model.eval()
    personal_feature_data = client_obj.local_data['X'][:, client_obj.n_fc_idx_end : client_obj.n_fc_idx_end + client_obj.n_fp]
    if personal_feature_data.numel() == 0: print("No personal feature data to plot."); return
    x_min, x_max = personal_feature_data.min().item(), personal_feature_data.max().item()
    x_range_tensor = torch.linspace(x_min, x_max, 200).unsqueeze(1)
    with torch.no_grad(): y_pred_ph = ph_model(x_range_tensor).cpu().numpy()
    plt.figure(figsize=(8,5))
    plt.plot(x_range_tensor.numpy().flatten(), y_pred_ph.flatten(),label=f'Client {client_obj.client_id} PH Learned Function', color='purple', linewidth=2.5)
    plt.title(title); plt.xlabel('Personalized Feature Value (Scaled)'); plt.ylabel('PH Model Output')
    plt.legend(frameon=True); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(f"{filename_base}_c{client_obj.client_id}.pdf", format='pdf', bbox_inches='tight')
    print(f"Saved: {filename_base}_c{client_obj.client_id}.pdf"); plt.show()

# --- Example call for 1D PH function viz (uncomment to use if NUM_FEATURES_PERSONAL == 1) ---
# if fed_its_hybrid_losses and NUM_FEATURES_PERSONAL == 1:
#     client_to_viz_1d_ph = server_hybrid.clients[0] 
#     visualize_1d_ph_function(client_to_viz_1d_ph, 
#                              title=f"Client {client_to_viz_1d_ph.client_id} Learned 1D PH Function",
#                              filename_base="ph_function")

print("\nSummary of Final Losses:")
if fed_its_hybrid_losses: print(f"Fed-ITS-Hybrid: {fed_its_hybrid_losses[-1]:.4f}")
else: print("Fed-ITS-Hybrid: N/A")
if fedavg_losses: print(f"FedAvg: {fedavg_losses[-1]:.4f}")
else: print("FedAvg: N/A")
if local_only_losses: print(f"Local Only (Structured PH): {local_only_losses[-1]:.4f}")
else: print("Local Only (Structured PH): N/A")

print("\nVisualizations generated (if data available). Check saved PDF files.")