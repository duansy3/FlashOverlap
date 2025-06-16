import torch

def div_up(x: int, y: int):
    return (x + y - 1) // y

def reorder_indices(S, hint):
    # Generate the original array of indices [0, 1, ..., S-1]
    original = list(range(S))
    
    # Create an empty list to store the new order of indices
    new_order = [-1] * S
    
    # Place the indices of the hint list in the first positions of the new order
    for i, element in enumerate(hint):
        new_order[element] = i
    
    # Place the remaining indices in the new order
    remaining_elements = [x for x in original if x not in hint]
    for i, element in enumerate(remaining_elements, start=len(hint)):
        new_order[element] = i
    
    return torch.tensor(new_order, dtype=torch.int, device="cuda")

def generate_row_mapping(
    M, N, BM, BN, S_list, world_size, device="cuda"
):
    total_tiles = (M * N) // (BM * BN)
    assert sum(S_list) == total_tiles, "sum(S_list) must equal total number of tiles"
    
    original_row_ids = torch.arange(M * N // BN, dtype=torch.int, device=device)
    reordered_row_id = torch.empty_like(original_row_ids)
    
    current_row = 0
    for S in S_list:
        chunk_size = S * BM
        chunk_row_ids = original_row_ids[current_row : current_row + chunk_size]
        
        # Compute row_id % world_size for the current chunk
        mod_values = chunk_row_ids % world_size
        
        # Sort the chunk based on mod_values (stable sort)
        _, sorted_indices = torch.sort(mod_values, stable=True)
        reordered_chunk = chunk_row_ids[sorted_indices]
        
        reordered_row_id[current_row : current_row + chunk_size] = reordered_chunk
        current_row += chunk_size
    
    # Compute remap: remap[original_row_id] = new_row_id
    remap = torch.empty_like(original_row_ids)
    remap[reordered_row_id] = torch.arange(len(reordered_row_id), dtype=torch.int, device=device)
    
    return remap

def reorder_rows_by_world_size(tensor, world_size):
    M, N = tensor.shape
    row_ids = torch.arange(M, device=tensor.device)  # [0, 1, 2, ..., M-1]
    mod_values = row_ids % world_size  # [0, 1, 2, ..., world_size-1, 0, 1, ...]

    # Get the indices that would sort mod_values (stable sort to preserve original order within groups)
    sorted_indices = torch.argsort(mod_values, stable=True)  # [indices of rows in new order]

    # Reorder the tensor
    reordered_tensor = tensor[sorted_indices]

    return reordered_tensor