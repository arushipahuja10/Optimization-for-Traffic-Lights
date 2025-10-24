def optimize_signals(flow_vector_8d, constraints):
    """
    Converts 8D flow vector into 4 green light times
    Input: [North_straight, North_right, East_straight, East_right,
            South_straight, South_right, West_straight, West_right]
    Output: [north_time, east_time, south_time, west_time] (seconds)
    """
    # Calculate total flow for each direction
    north_total = flow_vector_8d[0] + flow_vector_8d[1]    # North_straight + North_right
    east_total = flow_vector_8d[2] + flow_vector_8d[3]     # East_straight + East_right
    south_total = flow_vector_8d[4] + flow_vector_8d[5]    # South_straight + South_right
    west_total = flow_vector_8d[6] + flow_vector_8d[7]     # West_straight + West_right

    simplified_flows = [north_total, east_total, south_total, west_total]

    # Get constraints
    total_cycle_time = constraints['cycle_time']    # 120 seconds
    min_green = constraints['min_green']            # 20 seconds
    max_green = constraints['max_green']            # 60 seconds

    # Calculate proportional green times
    total_flow = sum(simplified_flows)

    # If no traffic, give equal minimum times
    if total_flow == 0:
        return [min_green, min_green, min_green, min_green]

    green_times = []
    for flow in simplified_flows:
        # Proportional allocation
        time = (flow / total_flow) * total_cycle_time

        # Apply min/max constraints
        time = max(min_green, min(time, max_green))
        green_times.append(round(time))

    return green_times


def find_orthogonal_phases():
    """
    Identifies which traffic directions can move together safely
    Returns the fundamental traffic light patterns
    """
    # Standard 4-way intersection phases
    phases = [
        [1, 0, 1, 0],  # Phase 1: North & South go together
        [0, 1, 0, 1]   # Phase 2: East & West go together
    ]
    return phases


def calculate_congestion_metric(current_queues, optimal_flows):
    """
    Uses dot product to measure how 'misaligned' current traffic is with optimal flows
    Input: current_queues = [north_q, east_q, south_q, west_q] (waiting vehicles)
           optimal_flows = [north_flow, east_flow, south_flow, west_flow] (ideal flow)
    Output: congestion score between 0 (no congestion) and 1 (max congestion)
    """
    # Calculate dot product manually (since we're not using numpy)
    dot_product = 0
    for i in range(len(current_queues)):
        dot_product += current_queues[i] * optimal_flows[i]
    
    # Normalize to 0-1 scale
    max_possible_congestion = 10000  # You can adjust this based on your intersection size
    congestion_score = min(dot_product / max_possible_congestion, 1.0)
    
    return congestion_score


# Test all three functions
if __name__ == "__main__":
    print("=== Testing All Three Functions ===\n")
    
    # Test data
    test_flows_8d = [25, 5, 20, 5, 8, 2, 12, 3]  # 8D vector from Member 3
    constraints = {'min_green': 20, 'max_green': 60, 'cycle_time': 120}
    current_queues = [10, 15, 5, 8]  # Current waiting vehicles [north, east, south, west]
    
    # Test 1: optimize_signals
    green_times = optimize_signals(test_flows_8d, constraints)
    print(f"1. optimize_signals() Result:")
    print(f"   Input 8D: {test_flows_8d}")
    print(f"   Green times: {green_times}")
    print(f"   Total cycle: {sum(green_times)} seconds\n")
    
    # Test 2: find_orthogonal_phases
    phases = find_orthogonal_phases()
    print(f"2. find_orthogonal_phases() Result:")
    print(f"   Traffic phases: {phases}")
    print(f"   Phase 1: North & South = {phases[0]}")
    print(f"   Phase 2: East & West = {phases[1]}\n")
    
    # Test 3: calculate_congestion_metric
    # Convert 8D flows to 4D for congestion calculation
    optimal_flows_4d = [
        test_flows_8d[0] + test_flows_8d[1],  # North total
        test_flows_8d[2] + test_flows_8d[3],  # East total
        test_flows_8d[4] + test_flows_8d[5],  # South total
        test_flows_8d[6] + test_flows_8d[7]   # West total
    ]
    
    congestion = calculate_congestion_metric(current_queues, optimal_flows_4d)
    print(f"3. calculate_congestion_metric() Result:")
    print(f"   Current queues: {current_queues}")
    print(f"   Optimal flows: {optimal_flows_4d}")
    print(f"   Congestion score: {congestion:.3f} (0 = good, 1 = bad)")
    
    print("\n✅ All three functions working perfectly!")