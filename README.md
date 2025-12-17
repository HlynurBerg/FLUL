# DISCLAIMER: Code comments and installation instruction is written partially using AI

# Federated Learning Testing Environment with Flower

This is a complete testing environment for federated learning using the Flower framework. The setup includes a server and multiple clients that can participate in federated learning experiments.

## Features

- **Flower-based federated learning**: Uses the Flower framework for federated averaging
- **Multiple datasets**: Supports MNIST, CIFAR-10, Fashion-MNIST, plus custom ImageFolder datasets
- **Pluggable models**: Switch between the baseline SimpleNet and a deeper ResNet-18 (supports >32×32 inputs)
- **Configurable clients**: Easy to configure number of clients and their data partitions
- **Server-side evaluation**: The server evaluates the global model on a test set
- **PyTorch backend**: Uses PyTorch for neural network models
- **Contribution Database**: Tracks and stores all client contributions throughout the FL process


## Installation

### Step 1: Create Virtual Environment

**On Windows (PowerShell):**
```powershell
# Run the setup script
.\setup_env.ps1

# Or manually create the virtual environment
python -m venv venv
```

**On Windows (Command Prompt):**
```cmd
# Run the setup script
setup_env.bat

# Or manually create the virtual environment
python -m venv venv
```

**On Linux/Mac:**
```bash
# Run the setup script
chmod +x setup_env.sh
./setup_env.sh

# Or manually create the virtual environment
python3 -m venv venv
```

### Step 2: Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```
*If you get an execution policy error, run:*
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

Once the virtual environment is activated, install the required packages:
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

You should see `(venv)` at the beginning of your command prompt, indicating the virtual environment is active.

## Usage

### Quick Start

1. **Start the server** (in one terminal):
   ```bash
   python server.py --dataset MNIST --model simplenet
   ```
   Or using the app.py entry point:
   ```bash
   python app.py server --dataset MNIST --model simplenet
   ```

2. **Start clients** (in separate terminals):
   ```bash
   # Terminal 1 - Client 0
   python client.py --client-id 0 --num-clients 3 --dataset MNIST --model simplenet
   
   # Terminal 2 - Client 1
   python client.py --client-id 1 --num-clients 3 --dataset MNIST --model simplenet
   
   # Terminal 3 - Client 2
   python client.py --client-id 2 --num-clients 3 --dataset MNIST --model simplenet
   ```
   
   Or using the app.py entry point:
   ```bash
   python app.py client --client-id 0 --num-clients 3 --dataset MNIST --model simplenet
   python app.py client --client-id 1 --num-clients 3 --dataset MNIST --model simplenet
   python app.py client --client-id 2 --num-clients 3 --dataset MNIST --model simplenet
   ```

### Configuration Options

**Server (`server.py`)**:
- The server runs on `0.0.0.0:8080` by default
- Configured for 10 rounds of federated learning
- Uses Federated Averaging (FedAvg) strategy
- Minimum 2 clients required
- `--dataset`: Dataset to use (MNIST, CIFAR10, FashionMNIST, or CUSTOM) - must match client dataset
- `--model`: `simplenet` (default) or `resnet18` for deeper experiments
- `--dataset-path`: Root folder for custom datasets (expects `train/` and `val/`)
- `--img-size`, `--num-channels`, `--num-classes`: Override input/output dimensions when using custom or high-resolution datasets

**Client (`client.py`)**:
- `--client-id`: The ID of the client (0-indexed)
- `--num-clients`: Total number of clients (determines data partitioning)
- `--dataset`: Dataset to use (MNIST, CIFAR10, FashionMNIST, or CUSTOM)
- `--model`: Architecture matching the server (same default/options)
- Same `--dataset-path`/override flags as the server so large/custom datasets stay in sync

### Example: Running with Different Datasets

**MNIST** (default):
```bash
python client.py --client-id 0 --num-clients 3 --dataset MNIST
```

**CIFAR-10**:
```bash
python client.py --client-id 0 --num-clients 3 --dataset CIFAR10
```

**Fashion-MNIST**:
```bash
python client.py --client-id 0 --num-clients 3 --dataset FashionMNIST
```

**Custom high-resolution dataset**:
```bash
python client.py --client-id 0 --num-clients 4 \
  --dataset CUSTOM \
  --dataset-path ./data/my_dataset \
  --img-size 224 \
  --num-channels 3 \
  --num-classes 100 \
  --model resnet18
```

### Larger Models & External Checkpoints

- Start with `--model resnet18` on `--dataset CIFAR10` to benchmark a deeper architecture better suited for higher-resolution inputs.
- If you need to sideload checkpoints that cannot be downloaded programmatically, place them under `models/pretrained/` (see `models/README.md`) and update the training scripts to load them before federated rounds begin.
- For datasets with images larger than 32×32 (e.g., 224×224), pass `--img-size 224 --num-channels 3` and pair them with `--model resnet18` so the ImageNet-style stem is used automatically.

### Custom High-Resolution Datasets

You can plug in any dataset organized as an `ImageFolder` with `train/` and `val/` subdirectories:

```
my_dataset/
├── train/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/
```

Example launch:

```bash
python server.py \
  --dataset CUSTOM \
  --dataset-path ./data/my_dataset \
  --img-size 224 \
  --num-channels 3 \
  --num-classes 100 \
  --model resnet18
```

Make sure every client uses the same flags:

```bash
python client.py \
  --client-id 0 \
  --num-clients 4 \
  --dataset CUSTOM \
  --dataset-path ./data/my_dataset \
  --img-size 224 \
  --num-channels 3 \
  --num-classes 100 \
  --model resnet18
```

### Banana Ripeness Example (416×416 images)

Place the provided dataset under `./Banana_ripeness/` (already organized as `train/`, `valid/`, `test/` with class-named folders). Run with:

```bash
python server.py \
  --dataset CUSTOM \
  --dataset-path ./Banana_ripeness \
  --img-size 416 \
  --num-channels 3 \
  --num-classes 4 \
  --model resnet18
```

Then start each client with the same overrides, e.g.:

```bash
python client.py \
  --client-id 0 \
  --num-clients 4 \
  --dataset CUSTOM \
  --dataset-path ./Banana_ripeness \
  --img-size 416 \
  --num-channels 3 \
  --num-classes 4 \
  --model resnet18
```

The loader automatically accepts `valid/` or `val/` folders for validation and will fall back to `test/` if present.

Horizontal partitioning remains the default; pass `--partition-type vertical` if you still want feature-wise splits (requires `--img-size`/`--num-channels` for custom data).

## Project Structure

```
FLUL/
├── app.py                  # Main entry point for running server/client
├── server.py               # Flower server implementation
├── client.py               # Flower client implementation
├── model.py                # Neural network model definition
├── utils.py                # Utility functions for data loading and training
├── contributions_db.py     # Contribution database for tracking client contributions
├── view_contributions.py   # Script to view and analyze contributions
├── manage_withdrawals.py   # Tool to manage client withdrawals and recalculate models
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Contribution Database

The system automatically tracks and stores all client contributions during federated learning. Contributions are stored in the `contributions/` directory with the following structure:

```
contributions/
└── {dataset}/
    ├── statistics.json              # Overall statistics
    ├── round_0001/
    │   ├── client_0000/
    │   │   ├── round_0001_client_0000_{timestamp}_metadata.json
    │   │   └── round_0001_client_0000_{timestamp}_params.pkl
    │   ├── client_0001/
    │   │   └── ...
    │   └── round_0001_aggregated_{timestamp}_metadata.json
    └── round_0002/
        └── ...
```

### Viewing Contributions

Use the `view_contributions.py` script to analyze contributions:

```bash
# View all contributions
python view_contributions.py --dataset MNIST

# View statistics
python view_contributions.py --dataset MNIST --stats

# Filter by round
python view_contributions.py --dataset MNIST --round 1

# Filter by client
python view_contributions.py --dataset MNIST --client-id 0
```

### Contribution Data

Each contribution includes:
- **Model parameters**: Full model state (saved as pickle file)
- **Metadata**: JSON file with:
  - Round number
  - Client ID
  - Timestamp
  - Number of training samples
  - Training metrics (loss, accuracy)
  - Parameter information

The aggregated (global) models are also saved after each round with server-side evaluation metrics.

### Client Withdrawal System (Git-like History)

The system supports **client withdrawal** similar to git history, where contributions are preserved but can be excluded from future calculations:

- **Withdraw a client**: Mark a client as withdrawn (contributions remain, just marked)
- **View history**: See all contributions with withdrawal status (git log style)
- **Recalculate models**: Rebuild aggregated models excluding withdrawn clients
- **Restore clients**: Undo a withdrawal and restore contributions

**Usage:**
```bash
# Withdraw a client
python manage_withdrawals.py withdraw --client-id 0 --dataset MNIST --reason "Client requested withdrawal"

# View contribution history (git log style)
python manage_withdrawals.py history --dataset MNIST

# View history for specific client
python manage_withdrawals.py history --dataset MNIST --client-id 0

# List all withdrawn clients
python manage_withdrawals.py list --dataset MNIST

# Recalculate models excluding withdrawn clients
python manage_withdrawals.py recalculate --dataset MNIST

# Restore a withdrawn client
python manage_withdrawals.py restore --client-id 0 --dataset MNIST
```

The withdrawal system maintains a complete history (`withdrawals.json`) and allows you to:
- Track when clients withdrew
- See which rounds were affected
- Recalculate aggregated models without withdrawn contributions
- Restore clients if needed

## How It Works

1. **Server**: The server coordinates the federated learning process:
   - Initializes a global model
   - Selects clients for each round
   - Aggregates model updates using Federated Averaging
   - Evaluates the global model

2. **Clients**: Each client:
   - Receives the global model from the server
   - Trains on its local data partition
   - Sends model updates back to the server

3. **Federated Averaging**: The server averages the model parameters from selected clients to update the global model.

4. **Contribution Database**: All client contributions are automatically tracked and stored:
   - Each client's model parameters are saved before aggregation
   - Aggregated (global) models are saved after each round
   - Metadata includes timestamps, metrics, number of samples, and more
   - Contributions are organized by dataset and round number

## Customization

### Changing the Model Architecture

Edit `model.py` to modify the neural network architecture. The `SimpleNet` class can be replaced with any PyTorch model.

### Adjusting Training Parameters

Modify training parameters in `utils.py`:
- Learning rate in `train_epoch()` function
- Batch size (passed to `load_data()`)
- Number of training epochs per round

### Changing Server Strategy

Edit `server.py` to use different Flower strategies:
- `FedAvg`: Standard federated averaging
- `FedProx`: Adds proximal term
- `FedNova`: Normalizes client updates
- And more...

## Troubleshooting

**Port already in use**: If port 8080 is already in use, modify the port in `server.py` and update clients accordingly.

**CUDA not available**: The code automatically falls back to CPU if CUDA is not available. No action needed.

**Data download issues**: The datasets will be automatically downloaded on first run. Ensure you have internet connectivity.

## Next Steps

- Implement non-IID data distribution
- Add differential privacy
- Implement secure aggregation
- Add more sophisticated client selection strategies
- Implement personalized federated learning approaches
