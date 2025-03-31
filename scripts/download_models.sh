# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "The 'gdown' package is required but not installed."
    echo "Installing gdown..."
    pip install gdown
fi

# Create checkpoints directory if it doesn't exist
if [ ! -d "checkpoints" ]; then
    echo "Creating checkpoints directory..."
    mkdir -p checkpoints
else
    echo "Checkpoints directory already exists."
fi

# Google Drive folder ID from the shared link
FOLDER_ID="17Ua28dtT8AeVGnZC35EN99hFWxzYwcBl"

echo "Downloading checkpoint files from Google Drive..."
# Try to download the entire folder
gdown --folder https://drive.google.com/drive/folders/${FOLDER_ID} -O ./checkpoints/ --remaining-ok

# Check if the download was successful
if [ $? -ne 0 ]; then
    echo "Folder download failed. This can happen due to Google Drive limitations."
    echo "You may need to manually download the files from:"
    echo "https://drive.google.com/drive/folders/${FOLDER_ID}"
else
    echo "Successfully downloaded checkpoint files to the 'checkpoints' directory."
fi

echo "Download process completed."