def run_time(start_time, end_time, device=None):
    '''# model running time'''
    total_time = end_time - start_time
    print(f"The model took {total_time:.3f} seconds on device: {device}\n")
    return total_time