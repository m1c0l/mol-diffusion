# Design Notes

Technical details on the architecture, training, and design decisions behind mol-diffusion.

---

## Why Diffusion?

Earlier generative approaches (GANs, VAEs) either required an adversarial training game that frequently collapsed, or collapsed to a single average output. Diffusion reframes generation as a sequence of small, supervised denoising steps — many easy problems instead of one hard one.

The mathematical justification comes from non-equilibrium thermodynamics: if a forward noise process adds small Gaussian increments, the reverse process is also approximately Gaussian at each step, and a neural network can learn its parameters. Gaussians are closed under addition, so you can jump to any noise level t in a single operation during training without simulating all intermediate steps.

---

## From Images to Molecules

In image diffusion, the state is a pixel grid. Noise corrupts pixel values. The denoiser is a U-Net that predicts the added noise given a noisy image and timestep.

In molecular diffusion, the state is a set of atomic coordinates. The key differences:

| Images | Molecules |
|---|---|
| Fixed grid (512×512) | Variable number of atoms per molecule |
| No structural prior between pixels | Bond graph encodes which atoms are coupled |
| Any pixel arrangement is valid input | Bond lengths and angles must satisfy chemistry |
| U-Net denoiser | GNN denoiser (graph structure is the inductive bias) |

For 2D molecules, the forward process adds Gaussian noise to (x, y) coordinates:

```
coords_t = sqrt(ᾱ_t) * coords_0  +  sqrt(1 - ᾱ_t) * ε,   ε ~ N(0, I)
```

where ᾱ_t follows a cosine schedule from 1 (clean) to 0 (pure noise) over T steps.

---

## Physics Constraints

Valid molecules have:
- **Bond lengths** near a fixed normalized value (~1.0), with very low variance
- **Bond angles** clustering at discrete values determined by orbital hybridization:
  - ~60°: 3-membered rings (epoxides, cyclopropanes)
  - ~109.5°: sp3 carbon (tetrahedral), 5-membered rings
  - ~120°: sp2 carbon, aromatic rings (dominant in drug-like molecules)
  - ~180°: linear bonds (triple bonds)

These are the distributions the trained model should reproduce at inference.

---

## Internal Coordinate Diffusion

Instead of corrupting raw (x, y) positions, the model diffuses over internal coordinates — bond lengths and absolute bond angles per spanning-tree edge. This places the noise directly in the space that chemistry constraints are defined over.

A BFS spanning tree rooted at atom 0 covers N-1 edges for N atoms. For each spanning tree edge (parent → child):

- **length**: Euclidean distance between parent and child positions
- **angle**: absolute direction of the bond in the 2D plane (atan2), in radians

Noisy coordinates are reconstructed from noisy internals by sequential BFS placement:

```
coords[child] = coords[parent] + length * [cos(angle), sin(angle)]
```

Ring closure bonds (non-spanning-tree edges) are not diffused separately — their geometry is implied by the spanning tree.

Noise is scaled separately per coordinate type:
- Lengths: same schedule as Cartesian diffusion
- Angles: noise std scaled by π so full noise ≈ uniform over [−π, π]
- Root position (atom 0): plain Cartesian noise

---

## EGNN Architecture

### Why Equivariance

A naive GNN that sees raw (x, y) coordinates treats a ring rotated 30° as a completely different input. During training it sees many rotations of similar molecules and minimizes MSE by predicting the average — which is the centroid. This mean-collapse means atoms cluster at the center of mass rather than forming rings and chains.

An E(2)-equivariant network guarantees: if you rotate the input coordinates by θ, the output rotates by the same θ. The network can't average across orientations because rotating the input rotates the output — there is no orientation-neutral prediction to hedge toward.

### EGNN Layer

Each layer updates both node features h (invariant) and coordinates x (equivariant):

```
rel_ij   = x_i - x_j                              # equivariant displacement
dist²_ij = ||rel_ij||²                             # invariant scalar

m_ij  = φ_e(h_i, h_j, dist²_ij, edge_attr)        # invariant message
x_i  ← x_i + Σ_j rel_ij · φ_x(m_ij)             # equivariant coord update
h_i  ← φ_h(h_i, Σ_j m_ij)                        # invariant node update
```

The coordinate update is equivariant because `rel_ij` transforms correctly under rotation and `φ_x(m_ij)` is a scalar (invariant). Raw coordinates never enter h — only invariant quantities (atom type, timestep, pairwise distances) do.

The coordinate track starts at noisy_coords and is refined across layers. The final x is the x0 prediction (predicted clean coordinates).

### Full Architecture

```
h init:  atom_emb(32) + t_emb(128) → Linear → hidden(128)
x init:  noisy_coords (2,)

4 × EGNNLayer(hidden=128, edge_dim=4)
  each layer updates h (invariant) and x (equivariant) jointly

output:  final x → predicted x0 (N, 2)
```

~478k parameters. Trains in under an hour per 30 epochs on M3 MPS.

---

## Constraint Losses

Two auxiliary losses penalize geometry violations in the predicted x0:

**Bond length loss**: MSE between predicted bond lengths and 1.0 (normalized target). Deduplicates undirected edges (src < dst only).

**Angle constraint loss**: For each bond angle triple (i, center, k), penalizes distance to the nearest valid angle (60/90/109.5/120/135/150/180°). Uses `atan2(|cross|, dot)` instead of `acos(dot/(|a||b|))` — avoids backward NaN on Apple MPS when the cosine hits ±1.

Both losses are weighted by ᾱ_t (the noise level) so they contribute only when the x0 estimate is meaningful (low noise t), not when the model is predicting from near-pure noise.

---

## Training

```
--use_internal       diffuse in internal coordinate space, predict x0
--lambda_constraint  weight for bond length + angle constraint losses (0.5 works well)
--lr_epochs          T_max for cosine LR schedule, independent of --epochs (default 200)
                     prevents premature LR annealing during iterative resume training
--resume             load weights from checkpoint before training
```

Training objective (internal mode):
```python
loss = MSE(x0_pred, x0_target) + λ * ᾱ_mean * (bond_length_loss + angle_loss)
```

---

## Noise Schedule

Cosine schedule (Nichol & Dhariwal, 2021). Gentler than linear — less aggressive corruption at high t, which helps for small molecules where structure matters even at moderate noise.

T=200 steps used throughout (training and sampling must match).

---

## Connection to RFDiffusion

| This project | RFDiffusion |
|---|---|
| 2D (x, y) coordinates | 3D SE(3): position + orientation per residue |
| Small molecules, bond graph | Protein backbone, sequence graph |
| Gaussian noise on R² | Gaussian + IGSO(3) noise on SE(3) |
| EGNN (E(2)-equivariant) | RoseTTAFold (SE(3)-equivariant) |
| Internal coords (bond lengths + angles) | Torsion angle parameterization |
| ~10k training molecules | ~57k PDB structures |
| Conditions: scaffold, properties (planned) | Conditions: hotspots, symmetry, secondary structure |

The core loop — corrupt with noise, train a network to predict clean structure, sample by running the reverse process — is identical.

---

## Conditioning (Planned Extensions)

The unconditional model learns p(molecule). Conditioning extends this to p(molecule | constraint):

**Fixed substructure (scaffold)**: Clamp a subgraph's atoms back to their target positions after every denoising step. Free atoms denoise around the scaffold via message passing.

**Global molecular properties**: Encode MW, logP, ring count as a vector, inject as condition embedding at each layer.

**Binding site geometry**: Represent pocket as a point cloud with chemical features. Cross-attention lets each molecule atom attend to nearby pocket points — the SBDD setup equivalent to DiffSBDD.

**Classifier-free guidance**: Drop condition with ~20% probability during training. At inference, extrapolate between conditional and unconditional predictions to control constraint adherence vs. diversity.
