# utils/data_utils.py
from torch.utils.data.dataloader import default_collate

def skip_none_collate(batch):
    """Custom collate function to filter out None samples and handle None fields."""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        # Return an empty dict that can be checked in the training loop.
        return {}
    
    # Create a new batch with non-None values
    result = {}
    
    # Get all keys from the first sample
    keys = batch[0].keys()
    
    for key in keys:
        # Collect all non-None values for this key
        valid_values = [sample[key] for sample in batch if key in sample and sample[key] is not None]
        
        if valid_values:
            try:
                # Try to collate the valid values
                result[key] = default_collate(valid_values)
            except Exception as e:
                # If collation fails, just use the first valid value
                print(f"Warning: Could not collate values for key {key}, using first value. Error: {e}")
                result[key] = valid_values[0]
    
    return result