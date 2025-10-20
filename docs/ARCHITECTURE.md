┌─────────────────────────────────────────┐
│  Human Motion Prediction (GP)           │
│  - 18-DOF skeleton tracking             │
│  - 0.4s prediction horizon              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Learned Control Barrier Function       │
│  - Lipschitz-constrained NN (L ≤ 1.0)  │
│  - 4-layer architecture (38→128→64→32→1)│
│  - Spectral normalization               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Multi-Objective MPC (H=20, 50Hz)       │
│  - 6 objectives (throughput, safety...) │
│  - QP solver (OSQP)                     │
│  - CBF hard constraints                 │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Robot Control (UR5)                    │
└─────────────────────────────────────────┘