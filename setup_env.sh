#!/bin/bash

# --- Helper Functions for colored output ---
print_info() {
    echo -e "\\033[1;34m[INFO]\\033[0m $1"
}

print_success() {
    echo -e "\\033[1;32m[SUCCESS]\\033[0m $1"
}

print_warning() {
    echo -e "\\033[1;33m[WARNING]\\033[0m $1"
}

print_error() {
    echo -e "\\033[1;31m[ERROR]\\033[0m $1"
}

print_header() {
    echo ""
    echo -e "\\033[1;35m=========================================================\\033[0m"
    echo -e "\\033[1;35m=      Brain CT Hemorrhage Project Environment Setup      =\\033[0m"
    echo -e "\\033[1;35m=========================================================\\033[0m"
    echo ""
}
# --- End Helper Functions ---

# --- Main Script ---
print_header

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi
print_info "Conda installation found."

# Define environment name and Python version
ENV_NAME="streamlit_env"
PYTHON_VERSION="3.11"
print_info "Target Environment: '$ENV_NAME' with Python $PYTHON_VERSION"
echo ""

# --- Environment and Dependency Management ---

# Check if the environment is already active
if [ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then
    print_info "Environment '$ENV_NAME' is already active."
    print_info "Installing/updating dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        print_success "Dependencies installed/updated successfully."
    else
        print_error "Failed to install/update dependencies."
        exit 1
    fi
else
    # Environment is not active, check if it exists
    if conda env list | grep -E "^${ENV_NAME}\\s" &> /dev/null; then
        print_info "Conda environment '$ENV_NAME' already exists."
    else
        print_info "Conda environment '$ENV_NAME' does not exist. Creating it now..."
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
        if [ $? -eq 0 ]; then
            print_success "Environment '$ENV_NAME' created successfully."
        else
            print_error "Failed to create environment '$ENV_NAME'. Please check your conda installation."
            exit 1
        fi
    fi
    
    # Guide user to activate and install
    print_warning "Environment is not active. Please run the following commands to complete setup:"
    echo ""
    echo "    conda activate $ENV_NAME"
    echo "    pip install -r requirements.txt"
    echo ""
fi

# --- Directory Setup ---
MODELS_DIR="models"
if [ ! -d "$MODELS_DIR" ]; then
    print_info "Creating directory: $MODELS_DIR"
    mkdir "$MODELS_DIR"
else
    print_info "Directory '$MODELS_DIR' already exists."
fi

# --- Final Instructions ---
echo ""
echo -e "\\033[1;35m--------------------------------------------------------\\033[0m"
echo -e "\\033[1;32m🚀 Setup Process Finished 🚀\\033[0m"
print_info "To run the application, ensure your environment is active and execute:"
echo ""
echo "    streamlit run app.py"
echo ""
echo -e "\\033[1;35m--------------------------------------------------------\\033[0m" 