import os,sys, math, re, time, datetime,json, traceback
from functools import wraps
from itertools import combinations
from collections import Counter,defaultdict
import numpy as np


def handle_assertions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {"err":repr(str(e)),
                    "err_trace":repr(''.join(traceback.format_exception(None, e, e.__traceback__)))}
    return wrapper

class reqVarLoadError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class semanticError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
class solFormatError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
class DecisionVariableLoadingError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


@handle_assertions
def prob1_verify_func(data_dict, hypothesis_solution):
    # Extract parameters
    numclasses = data_dict["numclasses"]
    numoptions = data_dict["numoptions"]
    numcars = data_dict["numcars"]
    numberPerClass = data_dict["numberPerClass"]
    optionsRequired = data_dict["optionsRequired"]
    windowSize = data_dict["windowSize"]
    optMax = data_dict["optMax"]
    try:
        slot = hypothesis_solution["slot"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    # tolerent for 0 based and 1 based
    if min(slot)==0:
        slot = [i+1 for i in slot]
    # check 1: verify slot length and distribution
    assert len(slot) == numcars, f"Error: The length of slot should be {numcars}, but got {len(slot)}."
    slot_counter = Counter(slot)
    for class_idx in range(numclasses):
        expected_count = numberPerClass[class_idx]
        actual_count = slot_counter.get(class_idx + 1, 0)
        assert actual_count == expected_count, f"Error: Expected {expected_count} cars of class {class_idx + 1}, but got {actual_count}."
    
    # check 2: verify option distribution within windows
    for option_idx in range(numoptions):
        max_allowed = optMax[option_idx]
        window = windowSize[option_idx]
        for start in range(numcars - window + 1):
            end = start + window
            window_slice = slot[start:end]
            option_count = sum(optionsRequired[class_idx - 1][option_idx] for class_idx in window_slice)
            assert option_count <= max_allowed, (
                f"Error: Option {option_idx + 1} exceeds the limit of {max_allowed} in window "
                f"starting at position {start}. Found {option_count} cars requiring this option."
            )
    return "pass"

@handle_assertions
def prob2_verify_func(data_dict, hypothesis_solution):
    # Extract parameters
    S = data_dict["S"]
    t = data_dict["t"]
    n = data_dict["n"]
    d = data_dict["d"]
    try:
        template = hypothesis_solution["template"]
        pressings = hypothesis_solution["pressings"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    

    # check 1: verify dimensions of template and pressings
    assert len(template) == t, f"Error: The number of templates should be {t}, but got {len(template)}."
    assert all(len(row) == n for row in template), "Error: Each template should have slots for all variations."
    assert len(pressings) == t, f"Error: The number of pressings should be {t}, but got {len(pressings)}."

    # check 2: verify that the total slots in each template equals S
    for i, row in enumerate(template):
        total_slots = sum(row)
        assert total_slots <= S, f"Error: Template {i+1} should have {S} slots, but got {total_slots}."

    # check 3: verify that enough of each variation is produced
    for i in range(n):
        total_produced = sum(template[j][i] * pressings[j] for j in range(t))
        assert total_produced >= d[i], (
            f"Error: Variation {i+1} requires {d[i]} units, but only {total_produced} units are produced."
        )


    # check 4: re-calculate total production and surplus
    cal_total_production = sum(pressings)
    surplus = cal_total_production * S - sum(d)

    # Verify total production is within bounds
    llower = math.ceil(sum(d) / S)
    lupper = 2 * llower
    assert llower <= cal_total_production <= lupper, (
        f"Error: Total production should be between {llower} and {lupper}, but got {cal_total_production}."
    )

    # Verify that surplus is within acceptable bounds
    assert 0 <= surplus <= lupper - llower, (
        f"Error: Surplus should be between 0 and {lupper - llower}, but got {surplus}."
    )

    # compare with known optimal value for instance q1
    ref_opt_val = {"q1":418}
    sol_opt = "optimal" if cal_total_production==ref_opt_val['q1'] else "sat"
    return "pass", sol_opt


@handle_assertions
def prob7_verify_func(data_dict, hypothesis_solution):
    # Extract parameters
    n = data_dict["n"]
    try:
        series = hypothesis_solution["series"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    
    
    # check 1: verify that series is a permutation of {0, 1, ..., n-1}
    expected_set = set(range(n))
    series_set = set(series)
    assert series_set == expected_set, (
        f"Error: The series must be a permutation of {expected_set}, but got {series_set}."
    )

    # check 2: calculate the differences between consecutive elements
    differences = [abs(series[i+1] - series[i]) for i in range(n-1)]

    # verify that differences form a permutation of {1, 2, ..., n-1}
    expected_differences_set = set(range(1, n))
    differences_set = set(differences)
    assert differences_set == expected_differences_set, (
        f"Error: The differences between consecutive elements must be a permutation of {expected_differences_set}, but got {differences_set}."
    )
    
    return "pass"

@handle_assertions
def prob8_verify_func(data_dict, hypothesis_solution):
    # Extract parameters
    deck_width = data_dict["deck_width"]
    deck_length = data_dict["deck_length"]
    n_containers = data_dict["n_containers"]
    n_classes = data_dict["n_classes"]
    width = data_dict["width"]
    length = data_dict["length"]
    container_classes = data_dict["classes"]
    separation = data_dict["separation"]
    try:
        left = hypothesis_solution["left"]
        right = hypothesis_solution["right"]
        bottom = hypothesis_solution["bottom"]
        top = hypothesis_solution["top"]
        orientation = hypothesis_solution["orientation"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"


    # check 1: verify containers fit within the deck
    for c in range(n_containers):
        assert 0 <= left[c] <= deck_width, f"Error: Container {c} has left coordinate out of deck bounds."
        assert 0 <= right[c] <= deck_width, f"Error: Container {c} has right coordinate out of deck bounds."
        assert 0 <= bottom[c] <= deck_length, f"Error: Container {c} has bottom coordinate out of deck bounds."
        assert 0 <= top[c] <= deck_length, f"Error: Container {c} has top coordinate out of deck bounds."

    # check 2: verify container sizes based on orientation
    for c in range(n_containers):
        actual_width = width[c] if orientation[c] == 1 else length[c]
        actual_length = length[c] if orientation[c] == 1 else width[c]

        assert right[c] == left[c] + actual_width, (
            f"Error: Container {c} has incorrect right coordinate; expected {left[c] + actual_width}, got {right[c]}."
        )
        assert top[c] == bottom[c] + actual_length, (
            f"Error: Container {c} has incorrect top coordinate; expected {bottom[c] + actual_length}, got {top[c]}."
        )

     # check 3: verify that no overlap between containers
    for c in range(n_containers):
        for k in range(c + 1, n_containers):
            class_c = container_classes[c]
            class_k = container_classes[k]
            min_separation = separation[class_c][class_k]

            assert (
                left[c] >= right[k] + min_separation or
                right[c] + min_separation <= left[k] or
                bottom[c] >= top[k] + min_separation or
                top[c] + min_separation <= bottom[k]
            ), f"Error: Containers {c} and {k} overlap or violate separation constraints."


    return "pass"

######################################################
@handle_assertions
def prob34_verify_func(data_dict, hypothesis_solution):
    # Extract parameters
    n_suppliers = data_dict["n_suppliers"]
    n_stores = data_dict["n_stores"]
    building_cost = data_dict["building_cost"]
    capacity = data_dict["capacity"]
    cost_matrix = data_dict["cost_matrix"]
    try:
        total_cost = hypothesis_solution["total_cost"]
        supplier = hypothesis_solution["supplier"]
        cost = hypothesis_solution["cost"]
        open_warehouses = hypothesis_solution["open"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    
    # Verify the total cost
    calculated_building_cost = sum(building_cost for i in range(n_suppliers) if open_warehouses[i])
    calculated_supply_cost = sum(cost[i] for i in range(n_stores))
    calculated_total_cost = calculated_building_cost + calculated_supply_cost

    assert calculated_total_cost == total_cost, (
        f"Error: The calculated total cost ({calculated_total_cost}) does not match the provided total cost ({total_cost})."
    )

    # TODO: allow both 0/1 based indexing
    # Verify the warehouse assignment and supply cost
    for i in range(n_stores):
        assigned_warehouse = supplier[i] - 1  # ! Adjust for 0-based indexing
        assert 0 <= assigned_warehouse < n_suppliers, (
            f"Error: Store {i+1} is assigned to an invalid warehouse {supplier[i]}."
        )
        assert cost[i] == cost_matrix[i][assigned_warehouse], (
            f"Error: The cost for store {i+1} is incorrect. Expected {cost_matrix[i][assigned_warehouse]}, got {cost[i]}."
        )

    # Verify warehouse capacities and open status
    warehouse_usage = [0] * n_suppliers

    for i in range(n_stores):
        assigned_warehouse = supplier[i] - 1  # ! Adjust for 0-based indexing
        warehouse_usage[assigned_warehouse] += 1

    for i in range(n_suppliers):
        if warehouse_usage[i] > 0:
            assert open_warehouses[i], f"Error: Warehouse {i+1} is supplying stores but is marked as closed."
            assert warehouse_usage[i] <= capacity[i], (
                f"Error: Warehouse {i+1} exceeds its capacity. Capacity is {capacity[i]}, but it supplies {warehouse_usage[i]} stores."
            )
        else:
            assert not open_warehouses[i], f"Error: Warehouse {i+1} is marked as open but does not supply any stores."

    return "pass"


######################################################################
@handle_assertions
def prob3_verify_func(data_dict, hypothesis_solution):
    # Step 1: Extract parameters
    n = data_dict["n"]
    try:
        quasiGroup = hypothesis_solution["quasiGroup"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"

    # Step 2: Verify the quasigroup Latin square property, each value should
    #  occur exactly once in each row and col
    #  the result can be either 0 based or 1 based as long as the rules are followed
    quasiGroup = np.array(quasiGroup)
    if np.min(quasiGroup) == 1:
        quasiGroup-=1

    for i in range(n):
        row_elements = quasiGroup[i]
        col_elements = [quasiGroup[j][i] for j in range(n)]
        assert sorted(row_elements) == list(range(0,n)), (
            f"Error: Row {i} does not contain all elements from 0 to {n-1}."
        )
        assert sorted(col_elements) == list(range(0,n)), (
            f"Error: Column {i} does not contain all elements from 0 to {n-1}."
        )

    # Step 3: Verify the quasigroup property for QG3.m problems
    for i in range(n):
        for j in range(n):
            lhs = quasiGroup[quasiGroup[i][j]][quasiGroup[j][i]]
            assert lhs == i, (
                f"Error: The condition quasiGroup[quasiGroup[{i},{j}], quasiGroup[{j},{i}]] = {i} fails. Got {lhs} instead."
            )

    return "pass"

#############################
@handle_assertions
def prob6_verify_func(data_dict, hypothesis_solution):
    """
    Note:
    For a given set, the differences between every pair of marks must be distinct
    eg: [0, 1, 4, 6]
    """
    # Step 1: Extract parameters
    m = data_dict["m"]
    try:
        mark = hypothesis_solution["mark"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"

    # Check 1: verify that the first mark is at position 0
    assert mark[0] == 0, "Error: The first mark should be at position 0."

    # Check 2: Verify that the marks are in strictly increasing order
    for i in range(1, m):
        assert mark[i] > mark[i - 1], f"Error: Marks are not in strictly increasing order at index {i}."

    # Check 3: Verify that all differences between marks are distinct
    differences = []
    for i in range(m):
        for j in range(i + 1, m):
            differences.append(mark[j] - mark[i])

    assert len(differences) == len(set(differences)), "Error: The differences between marks are not distinct."

    # Check 5: symmetry breaking; Not required.
    # assert (mark[1] - mark[0]) < (mark[m - 1] - mark[m - 2]), (
    #     f"Error: The first difference {mark[1] - mark[0]} is not less than the last difference {mark[m - 1] - mark[m - 2]}."
    # )
    ref_opt_val = {"q1":3}
    sol_opt = "optimal" if mark[-1]==ref_opt_val['q1'] else "sat"
    return "pass", sol_opt

#################################
@handle_assertions
def prob10_verify_func(data_dict, hypothesis_solution):
    """
    Critical checkings:
    Valid Groupings: Each group in every round must consist of distinct golfers.
    Unique Pairings: No golfer should play with the same golfer in more than one round.
    """
    # Step 1: Extract parameters
    n_groups = data_dict["n_groups"]
    n_per_group = data_dict["n_per_group"]
    n_rounds = data_dict["n_rounds"]
    #
    n_golfers = n_groups * n_per_group
    try:
        schedule = hypothesis_solution["schedule"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    
    # TODO: specify indexing in problem description
    # both 1 and 0 based indexing should be allowed
    schedule = np.array(schedule)
    if np.min(schedule) == 0:
        schedule+=1

    # check output format
    if len(schedule) != n_rounds:
        raise solFormatError(f"Invalid solution format: The schedule must contain {n_rounds} rounds, but got {len(schedule)}.")
    
    # Check 2: solution format validation
    for round in schedule:
        if len(round) != n_groups:
            raise solFormatError(f"Invalid solution format: Each round must have {n_groups} groups, but got {len(round)}.")
        for group in round:
            if len(group) != n_per_group:
                raise solFormatError(f"Invalid solution format: Each group must have {n_per_group} golfers.")
            if any(golfer < 1 or golfer > n_golfers for golfer in group):
                return "Invalid solution: Golfer IDs must be between 1 and the total number of golfers."

    
    # Check 3: Verify that all groups within a round consist of distinct golfers [[a,b],[c,d]]
    for round_idx, round_groups in enumerate(schedule):
        all_golfers = [golfer for group in round_groups for golfer in group]
        assert len(all_golfers) == len(set(all_golfers)), (
            f"Error: Round {round_idx + 1} has duplicate golfers within groups."
        )

    # Check 4: Verify that no golfer plays with the same golfer more than once
    golfer_pairs = set()  # To store all unique pairs of golfers
    for round_idx, round in enumerate(schedule):
        for group in round:
            # For each group, check all pairs of golfers
            for golfer1, golfer2 in combinations(group, 2):
                if (golfer1, golfer2) in golfer_pairs or (golfer2, golfer1) in golfer_pairs:
                    raise semanticError(f"Constraint violated: Golfer pair ({golfer1}, {golfer2}) plays together more than once.")
                golfer_pairs.add((golfer1, golfer2))

    return "pass"



###################################################################
def check_block_consts(rules, sequence):
    """ Check if a given sequence satisfies the block constraints in rules """
    blocks = []
    current_block = 0
    for cell in sequence:
        if cell == 1:
            current_block += 1
        elif current_block > 0:
            blocks.append(current_block)
            current_block = 0
    if current_block > 0:
        blocks.append(current_block)
    
    # Remove leading zeros in rules (optional zeros can be ignored)
    expected_blocks = [rule for rule in rules if rule > 0]
    
    return blocks == expected_blocks

@handle_assertions
def prob12_verify_func(data_dict, hypothesis_solution):
    # Step 1: Extract parameters
    rows = data_dict["rows"]
    row_rule_len = data_dict["row_rule_len"]
    row_rules = data_dict["row_rules"]
    cols = data_dict["cols"]
    col_rule_len = data_dict["col_rule_len"]
    col_rules = data_dict["col_rules"]
    try:
        grid = hypothesis_solution["grid"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    # check if shape matches
    if len(grid) != rows or any(len(row) != cols for row in grid):
        raise solFormatError(f"Invalid solution format: The grid must be a 2D array of size ({rows} by {cols}), but got ({len(grid)},{len(grid[0])}).")
    # check if the values are in required range -> binary
    for row in grid:
        if any(cell not in [0, 1] for cell in row):
            raise solFormatError(f"Invalid solution: Each cell in the grid must be either 0 (unshaded) or 1 (shaded).")
    
    # Step 2: row and col constraints
    # Check row constraints
    for i in range(rows):
        if not check_block_consts(row_rules[i], grid[i]):
            raise semanticError(f"Constraint violated: Row {i + 1} does not satisfy the constraints.")

    # Check column constraints
    for j in range(cols):
        col_sequence = [grid[i][j] for i in range(rows)]
        if not check_block_consts(col_rules[j], col_sequence):
            raise semanticError(f"Constraint violated: Column {j + 1} does not satisfy the constraints.")

    return "pass"




#######################
def is_ship(cell):
    return cell in ['c', 'l', 'r', 't', 'b', 'm']

def identify_ship_type(input_str):
    if input_str == 'c':
        return 'submarine'
    
    # if input_str represents a complete horizontal ship (like 'lmmmr')
    if input_str.startswith('l') and input_str.endswith('r') and 'm' in input_str:
        ship_length = len(input_str)
        if ship_length == 2:
            return 'destroyer'
        elif ship_length == 3:
            return 'cruiser'
        elif ship_length == 4:
            return 'battleship'
    
    # if input_str represents a complete vertical ship (like 'tmmmb')
    if input_str.startswith('t') and input_str.endswith('b') and 'm' in input_str:
        ship_length = len(input_str)
        if ship_length == 2:
            return 'destroyer'
        elif ship_length == 3:
            return 'cruiser'
        elif ship_length == 4:
            return 'battleship'
    
    # if it's not a complete ship or mine, return 0
    return 0

@handle_assertions
def prob14_verify_func(data_dict, hypothesis_solution):
    # Step 1: Extract parameters
    width = data_dict["width"]
    height = data_dict["height"]
    maxship = data_dict["maxship"]
    hint = data_dict["hint"]
    rowsum = data_dict["rowsum"]
    colsum = data_dict["colsum"]
    ship = data_dict["ship"]
    try:
        grid = hypothesis_solution["grid"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    ###

    # Check 1: Verify ship placement
    for i in range(height):
        for j in range(width):
            cell = grid[i][j]
            # Submarine
            if cell == 'c':  
                assert (
                    all(grid[x][y] == '.' for x in range(max(0, i-1), min(height, i+2))
                        for y in range(max(0, j-1), min(width, j+2)) if (x, y) != (i, j))
                ), f"Error: Submarine at ({i}, {j}) is adjacent to another ship."
            # check the placement of other ships (l, r, t, b, m)
            if cell == 'l':  #left end of horizontal ship
                assert j+1 < width and grid[i][j+1] in ['r', 'm'], f"Error: Left end of ship at ({i}, {j}) is not followed by a valid right/middle part."
            if cell == 'r':  #right end of horizontal ship
                assert j-1 >= 0 and grid[i][j-1] in ['l', 'm'], f"Error: Right end of ship at ({i}, {j}) is not preceded by a valid left/middle part."
            if cell == 't':  #top end of vertical ship
                assert i+1 < height and grid[i+1][j] in ['b', 'm'], f"Error: Top end of ship at ({i}, {j}) is not followed by a valid bottom/middle part."
            if cell == 'b':  # bottom end of vertical ship
                assert i-1 >= 0 and grid[i-1][j] in ['t', 'm'], f"Error: Bottom end of ship at ({i}, {j}) is not preceded by a valid top/middle part."
            if cell == 'm':  # middle part of a ship
                assert (
                    (i-1 >= 0 and i+1 < height and grid[i-1][j] in ['t', 'm'] and grid[i+1][j] in ['b', 'm']) or
                    (j-1 >= 0 and j+1 < width and grid[i][j-1] in ['l', 'm'] and grid[i][j+1] in ['r', 'm'])
                ), f"Error: Middle part of ship at ({i}, {j}) is not surrounded by valid ship parts."


    # Check 2: Verify row sums
    for i in range(height):
        row_sum = sum(1 for cell in grid[i] if is_ship(cell))
        assert row_sum == rowsum[i], f"Error: Row {i+1} has {row_sum} ship parts, but expected {rowsum[i]}."

    # Check 3: Verify column sums
    for j in range(width):
        col_sum = sum(1 for i in range(height) if is_ship(grid[i][j]))
        assert col_sum == colsum[j], f"Error: Column {j+1} has {col_sum} ship parts, but expected {colsum[j]}."

    # Check 4: Verify ship counts
    # Count ships and compare with expected counts
    grid_np = np.array(grid)
    grid_horizon = [obj for obj in ''.join(grid_np.flatten()).split('.') if obj]
    grid_vertical = [obj for obj in ''.join(grid_np.T.flatten()).split('.') if obj]
    ship_counts = {'submarine': 0, 'destroyer': 0, 'cruiser': 0, 'battleship': 0}
    for obj in grid_horizon+grid_vertical:
        obj_type = identify_ship_type(obj)
        if obj_type in ship_counts:
            ship_counts[obj_type]+=1
    # submarine counted twice horizon and vertical
    ship_counts['submarine']/=2
    # Verify ship counts
    assert ship_counts['submarine'] == ship[0], f"Error: Expected {ship[0]} submarines, but found {ship_counts['submarine']}."
    assert ship_counts['destroyer'] == ship[1], f"Error: Expected {ship[1]} destroyers, but found {ship_counts['destroyer']}."
    assert ship_counts['cruiser'] == ship[2], f"Error: Expected {ship[2]} cruisers, but found {ship_counts['cruiser']}."
    assert ship_counts['battleship'] == ship[3], f"Error: Expected {ship[3]} battleships, but found {ship_counts['battleship']}."

    return "pass"

#######################################################################
@handle_assertions
def prob15_verify_func(data_dict, hypothesis_solution):
    """
    Critical checks:
    1. Ball->box assignments: Each ball is placed in exactly one of the `c` boxes.
    2. Valid triples: for each triple \( (x, y, z) \) where \( x + y = z \), not all three balls are placed in the same box.
    """
    # Step 1: Extract parameters
    n = data_dict["n"]
    c = data_dict["c"]
    try:
        box_assignment = hypothesis_solution["box"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    #
    if len(box_assignment) != n:
        raise solFormatError(f"Invalid solution format: The 'box' array must have {n} elements.")

    ## Check 1: Verify that each ball is assigned to a valid box (between 1 and c)
    for i in range(n):
        assert 1 <= box_assignment[i] <= c, f"Error: Ball {i+1} is assigned to an invalid box {box_assignment[i]}."

    ## Check 2: Verify that no triple (x, y, z) where x + y = z are all in the same box
    for x in range(1, n):
        for y in range(x + 1, n + 1 - x):  # ! ensure y > x and z <= n
            z = x + y
            if z <= n:
                assert not (box_assignment[x-1] == box_assignment[y-1] == box_assignment[z-1]), (
                    f"Error: Balls {x}, {y}, and {z} are all in the same box {box_assignment[x-1]}."
                )

    return "pass"

##########################################################
# Helper function to check if the transition is valid
def is_valid_transition(v_i, p_i, v_j, p_j, allowed):
    for allowed_tuple in allowed:
        if [v_i, p_i, v_j, p_j] == allowed_tuple:
            return True
    return False

@handle_assertions
def prob16_verify_func(data_dict, hypothesis_solution):
    # Step 1: Extract parameters
    n = data_dict["n"]
    r = data_dict["r"]
    ry = data_dict["ry"]
    g = data_dict["g"]
    y = data_dict["y"]
    allowed = data_dict["allowed"]
    try:
        V = hypothesis_solution["V"]
        P = hypothesis_solution["P"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    #
    if len(V) != n or len(P) != n:
        raise solFormatError(f"Invalid solution format: 'V':{len(V)} and 'P':{len(P)} arrays must have {n} elements.")


    # Check 1: Verify valid transitions for traffic lights
    for i in range(n):
        j = (1 + i) % 4  # Calculate j based on the modulus
        assert is_valid_transition(V[i], P[i], V[j], P[j], allowed), (
            f"Error: The transition (V{i+1}, P{i+1}, V{j+1}, P{j+1}) = ({V[i]}, {P[i]}, {V[j]}, {P[j]}) is not valid."
        )

    return "pass"

###################################################################
@handle_assertions
def prob18_verify_func(data_dict, hypothesis_solution):
    """
    critical checkings:
    1. Correct transition of states: The sequence of states leading from the initial state to one of the accepting states must represent valid transitions according to the provided `transition_fn`.
    2. Minimal transfers: The `cost` of water transfers should match the total number of transitions between states in the sequence.
    3. Goal state: The final state in the sequence must be one of the accepting states.
    """
    # Step 1: Extract parameters
    n_states = data_dict["n_states"]
    input_max = data_dict["input_max"]
    initial_state = data_dict["initial_state"]
    accepting_states = data_dict["accepting_states"]
    transition_fn = data_dict["transition_fn"]
    nodes = data_dict["nodes"]
    try:
        cost = hypothesis_solution["cost"]
        sequence_of_states = hypothesis_solution["sequence_of_states"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    ##
    if not isinstance(sequence_of_states, list):
        raise solFormatError(f"Invalid solution: 'sequence_of_states' must be an array, but got {type(sequence_of_states)}")
    # the reference code in csplib added a base cost of 2, otherwise, cost=len(sequence_of_states)-1 
    # **@ re-calculate the objective values from output solutions
    # if len(sequence_of_states) != cost:
    #     raise solFormatError(f"Invalid solution: Length of 'sequence_of_states' must be 'cost'. (refer to csplib reference code)")

    if sequence_of_states[0] != initial_state:
        raise solFormatError(f"Invalid solution: 'sequence_of_states' must start with the 'initial_state'.")

    if sequence_of_states[-1] not in accepting_states:
        raise solFormatError(f"Invalid solution: Final state must be one of the 'accepting_states'.")


    # verify the sequence of states follows the transition function
    for i in range(1, len(sequence_of_states)):
        prev_state = sequence_of_states[i - 1]
        curr_state = sequence_of_states[i]
        assert transition_fn[prev_state - 1][curr_state - 1] > 0, (
            f"Error: Invalid transition from state {prev_state} to {curr_state}."
        )

    # solution optimality checking against known optimal value of instance q1
    ref_opt_val = {"q1":7}
    sol_opt = "optimal" if cost<ref_opt_val['q1'] else "sat"
    return "pass", sol_opt

##################################################################################################
@handle_assertions
def prob19_verify_func(data_dict, hypothesis_solution):
    """
    Critical checks:
    1. Unique nums: Each number between 1 and \( n^2 \) appears exactly once in the magic square.
    2. Magic sum: The sum of the numbers in each row, column, and both main diagonals must be equal to the `magic_sum`.
    3. Magic square: The structure of the matrix should match the format of the magic square as specified in the solution format.
    """
    # Step 1: Extract parameters
    n = data_dict["n"]
    try:
        magic_sum = hypothesis_solution["magic_sum"]
        square = hypothesis_solution["square"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    ##

    # Check 1: Verify unique numbers
    flattened_square = [num for row in square for num in row]
    expected_numbers = set(range(1, n * n + 1))
    assert set(flattened_square) == expected_numbers, (
        f"Error: The square does not contain all numbers from 1 to {n*n} without repetition."
    )

    # Check 2: Verify magic sum for rows
    for i in range(n):
        row_sum = sum(square[i])
        assert row_sum == magic_sum, (
            f"Error: The sum of row {i+1} is {row_sum}, but it should be {magic_sum}."
        )

    # Check 3: Verify magic sum for columns
    for j in range(n):
        col_sum = sum(square[i][j] for i in range(n))
        assert col_sum == magic_sum, (
            f"Error: The sum of column {j+1} is {col_sum}, but it should be {magic_sum}."
        )

    # Step 5: Verify magic sum for main diagonal
    main_diagonal_sum = sum(square[i][i] for i in range(n))
    assert main_diagonal_sum == magic_sum, (
        f"Error: The sum of the main diagonal is {main_diagonal_sum}, but it should be {magic_sum}."
    )

    # Step 6: Verify magic sum for co-diagonal
    co_diagonal_sum = sum(square[i][n-i-1] for i in range(n))
    assert co_diagonal_sum == magic_sum, (
        f"Error: The sum of the co-diagonal is {co_diagonal_sum}, but it should be {magic_sum}."
    )

    return "pass"

###########################################################################
@handle_assertions
def prob22_verify_func(data_dict, hypothesis_solution):
    # Step 1: Extract parameters
    num_work = data_dict["num_work"]
    num_shifts = data_dict["num_shifts"]
    min_num_shifts = data_dict["min_num_shifts"]
    shifts = data_dict["shifts"]
    try:
        total_shifts = hypothesis_solution["total_shifts"]
        x = hypothesis_solution["x"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"

    ##
    # format checking
    if not isinstance(total_shifts, int) or total_shifts < 0 or total_shifts > num_shifts:
        return f"Invalid solution: 'total_shifts':{total_shifts} must be a non-negative integer not exceeding 'num_shifts':{num_shifts}."
    
    if not isinstance(x, list) or len(x) != num_shifts:
        return "Invalid solution: 'x' must be a list of length 'num_shifts'."
    
    if not all(shift in [0, 1] for shift in x):
        return "Invalid solution: All elements in 'x' must be 0 or 1."

    # Check 1: Verify that all tasks are covered exactly once
    tasks_covered = [0] * num_work
    for i in range(num_shifts):
        if x[i] == 1:
            for task in shifts[i]:
                tasks_covered[task] += 1

    for task in range(num_work):
        assert tasks_covered[task] == 1, (
            f"Error: Task {task} is covered {tasks_covered[task]} times, but it should be covered exactly once."
        )

    # # Check 2: Verify the total number of selected shifts
    calculated_total_shifts = sum(x)
    assert calculated_total_shifts >= min_num_shifts, (
        f"Error: The total shifts {calculated_total_shifts} is less than the minimum required shifts {min_num_shifts}."
    )

    # solution optimality checking against known optimal value of instance q1
    ref_opt_val = {"q1":7}
    sol_opt = "optimal" if calculated_total_shifts==ref_opt_val['q1'] else "sat"
    return "pass", sol_opt

######################################################################
@handle_assertions
def prob23_verify_func(data_dict=None, hypothesis_solution=None):
    """
    eg solution:
      A B C
     D E F G
    H I J K L
     M N O P
      Q R S

    Note:
    1. All numbers are unique: The numbers 1 through 19 must be used exactly once in the hexagonal arrangement.
    2. Magic sum of 38: All rows, columns, and diagonals must sum to 38 (as specified in the problem description).
    """
    # Extract the solution format
    try:
        LD = hypothesis_solution["LD"]
        [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s] = LD
    except Exception as e:
        return f"dvarLoadError: {str(e)}"

    if not isinstance(LD, list) or len(LD) != 19:
        return "Invalid input: 'LD' must be a list of 19 integers."
    if not all(isinstance(num, int) for num in LD):
        return "Invalid input: All elements in 'LD' must be integers."
    if not all(1 <= num <= 19 for num in LD):
        return "Invalid input: All numbers must be between 1 and 19."
    if len(set(LD)) != 19:
        return "Constraint violated: All numbers in 'LD' must be unique."
    #########
    # Check 1: Verify the sums of all rows, columns, and diagonals equal 38
    assert a + b + c == 38, "Error: a + b + c does not equal 38."
    assert d + e + f + g == 38, "Error: d + e + f + g does not equal 38."
    assert h + i + j + k + l == 38, "Error: h + i + j + k + l does not equal 38."
    assert m + n + o + p == 38, "Error: m + n + o + p does not equal 38."
    assert q + r + s == 38, "Error: q + r + s does not equal 38."
    
    assert a + d + h == 38, "Error: a + d + h does not equal 38."
    assert b + e + i + m == 38, "Error: b + e + i + m does not equal 38."
    assert c + f + j + n + q == 38, "Error: c + f + j + n + q does not equal 38."
    assert g + k + o + r == 38, "Error: g + k + o + r does not equal 38."
    assert l + p + s == 38, "Error: l + p + s does not equal 38."
    
    assert c + g + l == 38, "Error: c + g + l does not equal 38."
    assert b + f + k + p == 38, "Error: b + f + k + p does not equal 38."
    assert a + e + j + o + s == 38, "Error: a + e + j + o + s does not equal 38."
    assert d + i + n + r == 38, "Error: d + i + n + r does not equal 38."
    assert h + m + q == 38, "Error: h + m + q does not equal 38."
    ###
  
    return "pass"

#########################################################################
@handle_assertions
def prob24_verify_func(data_dict, hypothesis_solution):
    # Step 1: Input validation
    n = data_dict['n']  # number of distinct numbers in each set
    k = data_dict['k']  # number of sets
    try:
        solution = hypothesis_solution["solution"]  # array representing the solution sequence
    except Exception as e:
        return f"dvarLoadError: {str(e)}"

    ## formatting
    # Check that all elements are integers between 1 and n
    if not all(isinstance(num, int) and 1 <= num <= n for num in solution):
        return f"Invalid solution: All numbers must be integers between 1 and {n}."

    # Check 1: Count occurrences
    occurrences = defaultdict(list)
    for idx, num in enumerate(solution):
        occurrences[num].append(idx)

    # Check that each number appears exactly k times
    for num in range(1, n + 1):
        if len(occurrences[num]) != k:
            return f"Constraint violated: Number {num} appears {len(occurrences[num])} times (should be {k} times)."

    # Check 2: Verify spacing constraints
    for num in range(1, n + 1):
        positions = sorted(occurrences[num])
        required_gap = num
        for i in range(len(positions) - 1):
            actual_gap = positions[i + 1] - positions[i] - 1
            if actual_gap != required_gap:
                return (f"Constraint violated: Number {num} occurrences are not {required_gap} positions apart, occured at pos ({positions[i]}, {positions[i + 1]}) with {actual_gap} pos apart). "
                        )

    # Check 3: Verify unique positions
    #  already ensured since we iterated over the sequence once

    return "pass"


###########################################################################
@handle_assertions
def prob28_verify_func(data_dict, hypothesis_solution):
    # Load data and extract matrix m
    v = data_dict['v']
    k = data_dict['k']
    lambda_val = data_dict['lambda_val']
    #
    try:
        m = hypothesis_solution["m"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    ##
    # Calculate b (number of blocks) and r (number of blocks each object occurs in)
    b = (lambda_val * v * (v - 1)) // (k * (k - 1))
    r = (lambda_val * (v - 1)) // (k - 1)
    # Validate matrix dimensions
    if len(m) != v:
        return f"Invalid solution: The matrix 'm' must have {v} rows."
    if any(len(row) != b for row in m):
        return f"Invalid solution: Each row in 'm' must have {b} columns."

    # Check that all elements are 0 or 1
    for i in range(v):
        for j in range(b):
            if m[i][j] not in [0, 1]:
                return f"Invalid solution: m[{i}][{j}] must be 0 or 1, but got {m[i][j]}."
            
    ## verify row sum constraint, each object occurs in exactly r blocks
    for i in range(v):
        row_sum = sum(m[i])
        assert row_sum == r, f"Error: Row {i+1} has {row_sum} ones, but should have {r} ones."
    ## verify column sum constraint, each block contains exactly k objects
    for j in range(b):
        column_sum = sum(m[i][j] for i in range(len(m)))
        assert column_sum == k, f"Error: Column {j+1} has {column_sum} ones, but should have {k} ones."
    ## Verify dot product constraint, any pair of distinct objects occurs together in exactly lambda blocks
    for i in range(v):
        for j in range(i + 1, v):
            dot_product = sum(m[i][col] & m[j][col] for col in range(len(m[i])))
            assert dot_product == lambda_val, f"Error: Rows {i+1} and {j+1} have a dot product of {dot_product}, but it should be {lambda_val}."

    return "pass"



#############################################################

def count_live_neighbors(grid, r, c):
    """
    [(-1, -1), (-1, 0), (-1, 1),
    (0, -1),         (0, 1),
    (1, -1),  (1, 0),  (1, 1)]
    """
    return (
        grid[r-1][c-1] + grid[r-1][c] + grid[r-1][c+1] +
        grid[r][c-1] + grid[r][c+1] +
        grid[r+1][c-1] + grid[r+1][c] + grid[r+1][c+1]
    )
@handle_assertions
def prob32_verify_func(data_dict, hypothesis_solution):
    # Extract variables from solution and data
    grid_size = data_dict["size"]
    size  = grid_size - 4
    try:
        grid = hypothesis_solution["grid"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    ###########
    # **format checker coverage
    if len(grid) != grid_size:
        return f"Invalid input: The grid must have {grid_size} rows but got {len(grid)}."

    for row_idx, row in enumerate(grid):
        if len(row) != grid_size:
            return f"Invalid input: Row {row_idx} must have {grid_size} columns."
        for col_idx, cell in enumerate(row):
            if cell not in [0, 1]:
                return f"Invalid input: Cell ({row_idx}, {col_idx}) must be 0 or 1."
    ##############

    #  Boundary conditions
    for i in range(grid_size):
        # First two and last two rows
        if grid[0][i] != 0 or grid[1][i] != 0 or grid[-2][i] != 0 or grid[-1][i] != 0:
            return f"Constraint violated: Cells in the first and last two rows must be 0."

        # First two and last two columns
        if grid[i][0] != 0 or grid[i][1] != 0 or grid[i][-2] != 0 or grid[i][-1] != 0:
            return f"Constraint violated: Cells in the first and last two columns must be 0."
        
    # #  Verify grid dimensions
    # For each cell in the central size x size grid -> (size-4)x(size-4):
    # If the cell is dead, check that it has exactly three live neighbors.
    # If the cell is alive, check that it has 2 or 3 live neighbors.
    for r in range(2, size + 2):
        for c in range(2, size + 2):
            live_neighbors = count_live_neighbors(grid, r, c)
            if grid[r][c] == 1:
                # Live cell: must have 2 or 3 live neighbors
                assert live_neighbors in [2,3], f"Error: Live cell at ({r}, {c}) has {live_neighbors} not in [2,3] live neighbors."
            else:
                # Dead cell: must have exactly 3 live neighbors to be alive in the next state
                assert live_neighbors != 3, f"Error: Dead cell at ({r}, {c}) has exactly 3 live neighbors but is not alive."

    # solution optimality checking against known optimal value of instance q1
    # objective value
    z=np.array(grid).sum()
    ref_opt_val = {"q1":6}
    sol_opt = "optimal" if z==ref_opt_val['q1'] else "sat"
    return "pass", sol_opt


####################################################################################
@handle_assertions
def prob34_verify_func(data_dict, hypothesis_solution):
    """
    Critial checkings:
    - The sum of the building costs for the open warehouses and the supply costs for the stores must equal the `total_cost`.
    - Each store must be assigned to exactly one open warehouse.
    - Each warehouse must not exceed its capacity.
    - Warehouses that are assigned at least one store must be marked as open.
    """
    # Extract variables from solution and data
    n_suppliers = data_dict["n_suppliers"]
    n_stores = data_dict["n_stores"]
    building_cost = data_dict["building_cost"]
    capacity = data_dict["capacity"]
    cost_matrix = data_dict["cost_matrix"]

    try:
        total_cost = hypothesis_solution["total_cost"]
        supplier = hypothesis_solution["supplier"]
        cost = hypothesis_solution["cost"]
        open_warehouses = hypothesis_solution["open"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    ###Solution Format Verification
    if not isinstance(total_cost, int) or total_cost < 0:
        raise AssertionError("Invalid solution: 'total_cost' must be a non-negative integer.")

    if not isinstance(supplier, list) or len(supplier) != n_stores:
        raise AssertionError("Invalid solution: 'supplier' must be a list of length 'n_stores'.")

    if not all(isinstance(s, int) and 1 <= s <= n_suppliers for s in supplier):
        raise AssertionError(f"Invalid solution: Each element in 'supplier' must be an integer between 1 and 'n_suppliers' ({n_suppliers}).")

    if not isinstance(cost, list) or len(cost) != n_stores:
        raise AssertionError("Invalid solution: 'cost' must be a list of length 'n_stores'.")

    if not all(isinstance(c, int) and c >= 0 for c in cost):
        raise AssertionError("Invalid solution: All elements in 'cost' must be non-negative integers.")

    if not isinstance(open_warehouses, list) or len(open_warehouses) != n_suppliers:
        raise AssertionError("Invalid solution: 'open' must be a list of length 'n_suppliers'.")

    if not all(o in [0,1] for o in open_warehouses):
        raise AssertionError("Invalid solution: All elements in 'open' must be binary values.")
    
    # Check 1: Verify total cost
    # The total cost consists of the sum of the building costs for the open warehouses and the 
    # supply costs for the stores. We need to verify that this sum matches the total_cost provided.
    # Calculate the building costs for open warehouses
    calculated_building_cost = sum(building_cost for i in range(n_suppliers) if open_warehouses[i])
    # Calculate the total supply costs for each store
    calculated_supply_cost = sum(cost[i] for i in range(n_stores))
    # Total cost should match the sum of building and supply costs
    # **@ re-calculate the objective values from output solutions
    calculated_total_cost = calculated_building_cost + calculated_supply_cost
    # assert calculated_total_cost == total_cost, (
    #     f"Error: Calculated total cost {calculated_total_cost} does not match the provided total cost {total_cost}."
    # )
    # open warehouse
    # Count how many stores are assigned to each warehouse
    warehouse_usage = [0] * n_suppliers
    for assigned_warehouse in supplier:
        warehouse_usage[assigned_warehouse - 1] += 1  # Convert to 0-indexed
    # Check that a warehouse is open if and only if it has stores assigned
    for i in range(n_suppliers):
        assert open_warehouses[i] == (warehouse_usage[i] > 0), (
            f"Error: Warehouse {i+1} is marked {'open' if open_warehouses[i] else 'closed'}, but it has {warehouse_usage[i]} stores assigned."
        )
    # Capacity
    # Check that no warehouse exceeds its capacity
    for i in range(n_suppliers):
        assert warehouse_usage[i] <= capacity[i], (
            f"Error: Warehouse {i+1} exceeds its capacity. Assigned {warehouse_usage[i]} stores, but capacity is {capacity[i]}."
        )

    # Check 3: Verify Store Assignments
    # Ensure that each store is assigned to exactly one warehouse and the assignment is 
    # reflected correctly in the cost.
    for i in range(n_stores):
        assigned_warehouse = supplier[i] - 1  # Convert to 0-indexed
        expected_cost = cost_matrix[i][assigned_warehouse]
        assert cost[i] == expected_cost, (
            f"Error: Store {i+1} has a supply cost of {cost[i]}, but expected cost is {expected_cost}."
        )

    # solution optimality checking against known optimal value of instance q1
    # objective
    # obj_val = total_cost
    ref_opt_val = {"q1":383}
    sol_opt = "optimal" if calculated_total_cost==ref_opt_val['q1'] else "sat"
    return "pass", sol_opt

##############################################################
def calculate_waiting_time(optimal_order, duration, rehearsal, num_pieces, num_players):
    # Initialize waiting time for all players
    waiting_times = [0] * num_players
    
    # Calculate the waiting time for each player
    for p in range(num_players):
        first_piece, last_piece = None, None
        # Find the first and last pieces the player is involved in
        for i in range(num_pieces):
            if rehearsal[p][optimal_order[i] - 1] == 1:  # ! convert to 0-indexed
                if first_piece is None:
                    first_piece = i
                last_piece = i
        
        # Calculate the waiting time for the player
        if first_piece is not None and last_piece is not None:
            waiting_time = 0
            for i in range(first_piece, last_piece + 1):
                if rehearsal[p][optimal_order[i] - 1] == 0:
                    waiting_time += duration[optimal_order[i] - 1]
            waiting_times[p] = waiting_time
    
    return waiting_times

@handle_assertions
def prob39_verify_func(data_dict, hypothesis_solution):
    # Extract input data and Extract hypothesis solution
    num_pieces = data_dict["num_pieces"] 
    num_players = data_dict["num_players"]
    duration = data_dict["duration"]
    rehearsal = data_dict["rehearsal"]
    
    try:
        order = hypothesis_solution["order"]
        # Get the provided total waiting time from the hypothesis solution
        total_waiting_time = hypothesis_solution["total_waiting_time"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    ##
    if not isinstance(order, list) or len(order) != num_pieces:
        return "Invalid solution: 'order' must be a list of length 'num_pieces'."

    if set(order) != set(range(1, num_pieces + 1)):
        return "Invalid solution: 'order' must be a permutation of numbers from 1 to 'num_pieces'."
    
    # **@ re-calculate the objective values from output solutions
    # Calculate waiting time for each player
    waiting_times = calculate_waiting_time(order, duration, rehearsal, num_pieces, num_players)
    
    
    # Calculate total waiting time
    calculated_total_waiting_time = sum(waiting_times)
    
    # Verify if the calculated total waiting time matches the hypothesis solution
    # assert calculated_total_waiting_time == total_waiting_time, (
    #     f"Error: The calculated total waiting time {calculated_total_waiting_time} != reported {total_waiting_time}."
    # )

    # solution optimality checking against known optimal value of instance q1
    # print(calculated_total_waiting_time)
    ref_opt_val = {"q1":0}
    sol_opt = "optimal" if calculated_total_waiting_time==ref_opt_val['q1'] else "sat"
    return "pass", sol_opt


###########################################################################
@handle_assertions
def prob41_verify_func(data_dict=None, hypothesis_solution=None):
    # Extract input data and Extract hypothesis solution
    try:
        vars = hypothesis_solution["vars"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    ##
    if not isinstance(vars, list) or len(vars) != 9:
        return f"Invalid input: 'vars' must be a list of 9 integers, but got {vars}."
    
    if not all(isinstance(v, int) and 1 <= v <= 9 for v in vars):
        return f"Invalid input: All variables must be integers between 1 and 9, but got {vars}."
    
    if len(set(vars)) != 9:
        return f"Constraint violated: All variables must be distinct digits, but got {vars}."
    ##
    A, B, C, D, E, F, G, H, I = vars
    D1 = 10 * B + C # BC
    D2 = 10 * E + F # EF
    D3 = 10 * H + I # HI
    # Ensure D1, D2, D3 are between 1 and 81 (since variables are digits from 1 to 9)
    if not (1 <= D1 <= 99 and 1 <= D2 <= 99 and 1 <= D3 <= 99):
        return "Invalid values: D1, D2, D3 must be between 1 and 99."
    # this is equivalent to original equation, both sides mult by D1*D2*D3
    left_side = A * D2 * D3 + D * D1 * D3 + G * D1 * D2
    right_side = D1 * D2 * D3
    if left_side != right_side:
        return f"Constraint violated: The main equation is not satisfied. Left side {left_side}, Right side {right_side}."
    #
    sol_opt=None
    return "pass",sol_opt

#####################################################################################
@handle_assertions
def prob44_verify_func(data_dict=None, hypothesis_solution=None):
    # Extract input data and Extract hypothesis solution
    N = data_dict["N"]
    try:
        sets = hypothesis_solution["sets"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    
    b_expected = N * (N - 1) // 6  # Expected number of blocks
    if len(sets) != b_expected:
        return f"Invalid solution: Number of blocks should be {b_expected}, but got {len(sets)}."
    
    # Check 1: Check Block Size and Element Validity
    elements = set(range(1, N + 1))
    for idx, block in enumerate(sets):
        if not isinstance(block, list) or len(block) != 3:
            return f"Constraint violated: Block {idx + 1} does not contain exactly 3 elements."
        if len(set(block)) != 3:
            return f"Constraint violated: Block {idx + 1} contains duplicate elements."
        if not all(isinstance(elem, int) and elem in elements for elem in block):
            return f"Constraint violated: Block {idx + 1} contains invalid elements. Elements must be integers between 1 and {N}."
        
    # Check 2: pairwise occurrence verify, every pair must occur exactly once
    pair_count = defaultdict(int)
    for block in sets:
        for pair in combinations(block, 2):
            pair = tuple(pair) if pair[0] < pair[1] else tuple(reversed(pair))
            pair_count[pair] += 1
    # check num of unique pair = all combo of pairs
    total_pairs = N * (N - 1) // 2  # total number of pairs
    if len(pair_count) != total_pairs:
        return f"Constraint violated: Number of unique pairs is {len(pair_count)}, expected {total_pairs}."
    for pair, count in pair_count.items():
        if count != 1:
            return f"Constraint violated: Pair {pair} occurs {count} times, but should occur exactly once."
        
    # Check 3: Intersection Constraint Verification
    # Since we have already ensured that each pair occurs exactly once,
    # any two blocks can share at most one element. However, we'll verify this explicitly.
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            intersection = set(sets[i]).intersection(sets[j])
            if len(intersection) > 1:
                return f"Constraint violated: Blocks {i + 1} and {j + 1} share more than one common element."
    #
    sol_opt=None
    return "pass",sol_opt

#######################################################################################################
@handle_assertions
def prob49_verify_func(data_dict, hypothesis_solution):
    # Load parameters and extract solutiions
    n = data_dict["n"]
    try:
        res = hypothesis_solution["res"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    
    if not isinstance(res, list) or len(res) != 2:
        return "Invalid solution format: 'res' must be a list containing two sublists."
    A, B = res
    # Verify that A and B are lists
    if not isinstance(A, list) or not isinstance(B, list):
        return "Invalid solution format: Both subsets must be lists."
    
    # verify partition
    elements = set(range(1, n + 1))
    set_A = set(A)
    set_B = set(B)

    if not set_A.isdisjoint(set_B):
        return "Constraint violated: Subsets A and B must be disjoint."

    if set_A.union(set_B) != elements:
        return "Constraint violated: Subsets A and B must contain all numbers from 1 to n exactly once."

    # check equal Cardinality
    if len(A) != len(B):
        return "Constraint violated: Subsets A and B must have the same number of elements."

    # Check equal Sum
    sum_A = sum(A)
    sum_B = sum(B)

    if sum_A != sum_B:
        return f"Constraint violated: Sum of A ({sum_A}) does not equal sum of B ({sum_B})."

    # Check equal Sum of Squares
    sum_squares_A = sum(x**2 for x in A)
    sum_squares_B = sum(x**2 for x in B)

    if sum_squares_A != sum_squares_B:
        return f"Constraint violated: Sum of squares of A ({sum_squares_A}) does not equal sum of squares of B ({sum_squares_B})."
    #
    sol_opt=None
    return "pass",sol_opt


#################################################################################
@handle_assertions
def prob50_verify_func(data_dict, hypothesis_solution):
    # Extract the number of vertices n from data_dict
    n = data_dict["n"]
    # Extract the degree sequence from hypothesis_solution
    try:
        degrees = hypothesis_solution["degrees"]
        x = hypothesis_solution["x"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    ### formatting
    if not isinstance(degrees, list) or len(degrees) != n:
        return f"Invalid input: 'degrees' must be a list of length {n}."

    if not all(isinstance(d, int) and d > 0 for d in degrees):
        return "Invalid input: All degrees must be positive integers."
    
    ## 
     # Check 1: Verify that the degrees are sorted in non-increasing order
    assert all(degrees[i] >= degrees[i+1] for i in range(len(degrees)-1)), (
        "Error: Degrees are not sorted in non-increasing order."
    )
    
    # Check 2: Verify that each degree is greater than 0 and is divisible by 3
    assert all(degree > 0 and degree % 3 == 0 for degree in degrees), (
        "Error: Each degree must be greater than 0 and divisible by 3."
    )
    
    # Check 3: Verify that the sum of degrees is divisible by 12
    sum_degrees = sum(degrees)
    assert sum_degrees % 12 == 0, (
        f"Error: The sum of degrees is {sum_degrees}, which is not divisible by 12."
    )

    # Verify the graph does not contain any diamonds
    for quartet in combinations(range(n), 4):
        i, j, k, l = quartet
        edge_sum = (x[i][j] + x[i][k] + x[i][l] +
                    x[j][k] + x[j][l] +
                    x[k][l])
        if edge_sum > 4:
            return (f"Constraint violated: Quartet of vertices {i + 1}, {j + 1}, {k + 1}, {l + 1} "
                    f"forms a diamond (edge sum {edge_sum} > 4).")

    # Check 5: Verify the degree of each vertex matches the sum of edges in the adjacency matrix
    for i in range(n):
        assert degrees[i] == sum(x[i]), (
            f"Error: Degree of vertex {i} does not match the adjacency matrix."
        )
    
    # Check 6: Verify no loops (no vertex is connected to itself), diagnal must be zeros
    assert all(x[i][i] == 0 for i in range(n)), "Error: Loops found (vertices connected to themselves)."
    
    # Check 7: Verify the graph is undirected (x[i][j] == x[j][i]), adjacency matrix must be symmetric at x[i][j] and x[j][i].
    assert all(x[i][j] == x[j][i] for i in range(n) for j in range(n)), "Error: Graph is not undirected."

    #
    sol_opt=None
    return "pass",sol_opt


###############################################################################
@handle_assertions
def prob53_verify_func(data_dict, hypothesis_solution):
    # Step 1: Extract the parameters from data_dict
    m = data_dict["m"]
    n = data_dict["n"]
    graph = data_dict["graph"]

    # Step 2: Extract the node labels and edge labels from output solution
    try:
        nodes = hypothesis_solution["nodes"]
        edges = hypothesis_solution["edges"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    
    # check for nodes
    # Check if all node labels are unique and in the range [0, m]
    if not isinstance(nodes, list) or len(nodes) != n:
        return f"Invalid solution: 'nodes':({type(nodes)}) must be a list of length 'n':{n}."
    assert len(set(nodes)) == n, "Error: Node labels are not unique."
    assert all(0 <= node <= m for node in nodes), "Error: Node labels are not in the range [0, m]."

    # check for edges
    # Check if all edge labels are unique and in the range [1, m]
    if not isinstance(edges, list) or len(edges) != m:
        return f"Invalid solution: 'edges':({type(edges)}) must be a list of length 'm': {m}."
    assert len(set(edges)) == m, "Error: Edge labels are not unique."
    assert all(1 <= edge <= m for edge in edges), "Error: Edge labels are not in the range [1, m]."

    # Verify the graceful labelling rule: |f(x) - f(y)| for each edge must be unique
    for i in range(m):
        u, v = graph[i]
        edge_label = abs(nodes[u - 1] - nodes[v - 1])  # get the edge label as the absolute difference of node labels
        assert edge_label == edges[i], (
            f"Error: Edge {i+1} label mismatch. Expected {edges[i]}, but got {edge_label}."
        )
    
    #
    sol_opt=None
    return "pass",sol_opt

####################################################################################################
@handle_assertions
def prob54_verify_func(data_dict, hypothesis_solution):
    # Extract the board size 'n' from data_dict
    n = data_dict["n"]
    # Extract the positions of queens from hypothesis_solution
    try:
        queens = hypothesis_solution["queens"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    
    # TODO: allow both 0/1 based indexing
    # both 1 based and 0 based indexing should be allowed
    queens = np.array(queens)
    if np.min(queens) == 0:
        queens+=1
    ##
    # Step 1: Check if all queens are in the range [1, n]
    assert len(queens) == n, f"Error: Number of queens {len(queens)} does not match the board size {n}."
    assert all(1 <= q <= n for q in queens), f"Error: Queen positions must be in the range [1, {n}]."
    assert len(set(queens)) ==n, f"Constraint violated: Exists queens share the same row."

    # Step 2: Check the rows and diagonals for conflicts
    for i in range(n):
        for j in range(i + 1, n):
            # Same row check, duplicated with len(set(queens)) ==n
            assert queens[i] != queens[j], f"Error: Queens at column {i + 1} and {j + 1} are in the same row."

            # Diagonal check
            assert abs(queens[i] - queens[j]) != abs(i - j), (
                f"Error: Queens at column {i + 1} and {j + 1} are on the same diagonal."
            )
    #
    sol_opt=None
    return "pass",sol_opt

###################################################################################################

@handle_assertions
def prob56_verify_func(data_dict, hypothesis_solution):
    # extract parameters, load solution variables
    r = data_dict["r"]
    n = data_dict["n"]
    demand = data_dict["demand"]
    capacity_nodes = data_dict["capacity_nodes"]
    try:
        z = hypothesis_solution["objective"]
        rings = hypothesis_solution["rings"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    ### check solution format
    if not isinstance(rings, list) or len(rings) != r:
        return "Invalid solution: 'rings' must be a list of length 'r'."
    for idx, ring in enumerate(rings):
        if not isinstance(ring, list) or len(ring) != n:
            return f"Invalid solution: 'rings[{idx}]' must be a list of length 'n'."
        for val in ring:
            if val not in [0, 1]:
                return f"Invalid solution: Ring values must be 0 or 1."
    
    # **@ re-calculate the objective values from output solutions
    # Chceck 1: Verify the total number of ADMs
    calculated_z = sum(rings[ring][client] for ring in range(r) for client in range(n))

    # Chceck 3: Verify that demand is satisfied, check that the demand between nodes is satisfied by at least one ring
    for client1 in range(n):
        for client2 in range(client1 + 1, n):
            if demand[client1][client2] == 1:
                demand_satisfied = any(rings[ring][client1] + rings[ring][client2] == 2 for ring in range(r))
                assert demand_satisfied, (
                    f"Error: Demand between nodes {client1} and {client2} is not satisfied."
                )

    # Chceck 4: Verify ring capacities
    for ring in range(r):
        num_nodes_on_ring = sum(rings[ring][client] for client in range(n))
        assert num_nodes_on_ring <= capacity_nodes[ring], (
            f"Error: Ring {ring} exceeds its capacity. It has {num_nodes_on_ring} nodes, but the capacity is {capacity_nodes[ring]}."
        )
    # solution optimality checking against known optimal value of instance q1
    ref_opt_val = {"q1":7}
    sol_opt = "optimal" if calculated_z==ref_opt_val['q1'] else "sat"
    return "pass", sol_opt
###################################################################################################
@handle_assertions
def prob57_verify_func(data_dict, hypothesis_solution):
    """
    Critical checks:
     1. Row unique: each row contains unique values from 1 to 9.
     2. Column unique: each column contains unique values from 1 to 9.
     3. 3x3 Box unique: each 3x3 box contains unique values from 1 to 9.
     4. Cage constraints: the sum of the values in each cage matches the provided sum, ensure that the values in the cage are unique.
    """
    # Extract the parameters from data_dict
    n = data_dict["n"]
    num_p = data_dict["num_p"]
    num_hints = data_dict["num_hints"]
    max_val = data_dict["max_val"]
    P = data_dict["P"]

    # Extract the solution variables
    try:
        grid = hypothesis_solution["grid"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    # check sol format
    if not isinstance(grid, list) or len(grid) != n:
        return f"Invalid solution: 'grid' must be a {n}x{n} list, recieved grid type:{type(grid)}"
    
    # Check 1: Verify each row has unique values
    for i in range(n):
        assert len(set(grid[i])) == n, f"Error: Row {i+1} does not contain unique values."

    # Check 2: Verify each column has unique values
    for j in range(n):
        col_values = [grid[i][j] for i in range(n)]
        assert len(set(col_values)) == n, f"Error: Column {j+1} does not contain unique values."

    # Check 3: Verify each 3x3 box has unique values
    for box_row in range(0, n, 3):
        for box_col in range(0, n, 3):
            box_values = [grid[i][j] for i in range(box_row, box_row+3) for j in range(box_col, box_col+3)]
            assert len(set(box_values)) == n, f"Error: 3x3 Box starting at ({box_row+1}, {box_col+1}) does not contain unique values."

    # Check 4: Verify the cage constraints
    for p in range(num_p):
        cage_sum = 0
        for hint_idx in range(num_hints):
            row = P[p][2*hint_idx] - 1  # ! 1-based to 0-based index
            col = P[p][2*hint_idx + 1] - 1
            if row >= 0 and col >= 0:
                cage_sum += grid[row][col]
        expected_sum = P[p][2*num_hints]
        assert cage_sum == expected_sum, f"Error: Cage {p+1} does not match expected sum {expected_sum}. Actual sum: {cage_sum}."
    #
    sol_opt=None
    return "pass",sol_opt

################################################################################################
@handle_assertions
def prob67_verify_func(data_dict=None, hypothesis_solution=None):
    """
    Critical checks:
     1. Latin square: The rows and columns must contain each number from 1 to `N` exactly once.
     2. Start position: The solution must respect the prefilled values in the `start` grid.
    """
    # Extract the parameters from data_dict
    N = data_dict["N"]
    start = data_dict["start"]

    # Extract the hypothesis solution
    try:
        puzzle = hypothesis_solution["puzzle"]
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    if not isinstance(puzzle, list) or len(puzzle) != N:
        return "Invalid solution: 'puzzle' must be a list of length N."
    for row in puzzle:
        if not isinstance(row, list) or len(row) != N:
            return "Invalid solution: Each row in 'puzzle' must have length N."

     # Check 1: Verify that the puzzle is a Latin square
     #  -> rows and columns must have all distinct values
    for i in range(N):
        row = puzzle[i]
        col = [puzzle[j][i] for j in range(N)]
        assert len(set(row)) == N, f"Error: Row {i+1} does not contain distinct values: {row}"
        assert len(set(col)) == N, f"Error: Column {i+1} does not contain distinct values: {col}"
        assert all((1 <= val <= N) for val in row), f"Invalid values in row {i+1}: All values must be between 1 and N."
        assert all((1 <= val <= N) for val in col), f"Invalid values in col {i+1}: All values must be between 1 and N."

    # Check 2: Verify that the filled values match the provided start positions
    for i in range(N):
        for j in range(N):
            if start[i][j] != 0:
                assert puzzle[i][j] == start[i][j], f"Error: Puzzle value at ({i+1},{j+1}) does not match start value. Expected {start[i][j]}, got {puzzle[i][j]}."
    #
    sol_opt=None
    return "pass",sol_opt

###############################################################################
@handle_assertions
def prob74_verify_func(data_dict, hypothesis_solution):
    """
    Critical checks:
     1. The vertices that are marked `True` in the solution must form a clique. This means all pairs of vertices in this set should be adjacent in the adjacency matrix.
     2. The actual size of the clique should match the size reported in the solution.
    """
    # Extract the parameters from data_dict
    n = data_dict["n"]
    adj = data_dict["adj"]

    # Extract the output solution variables
    try:
        c = hypothesis_solution["c"]  # True/False array representing clique membership
        size = hypothesis_solution["size"]  # Size of the clique
    except Exception as e:
        return f"dvarLoadError: {str(e)}"
    # Formatting
    if not isinstance(c, list) or len(c) != n:
        return "Invalid solution: 'c' must be a list of length 'n'."
    
    if not all(ci in [0, 1] for ci in c):
        return f"Invalid solution: All elements in 'c' must be integers values in [0,1], but got f{c}."

    # if not isinstance(size, int) or size < 0 or size > n:
    #     return "Invalid solution: 'size' must be an integer between 0 and 'n'."
    
    # **@ re-calculate the objective values from output solutions
    # Verify that the size is correct
    calculated_size = sum(c)
    # assert calculated_size == size, f"Error: Reported size {size} does not match the actual size {calculated_size}."
    # get the indices of vertices included in the clique
    selected_vertices  = [i for i, included in enumerate(c) if included]
    for i in range(len(selected_vertices)):
        for j in range(i+1, len(selected_vertices)):
            v1, v2 = selected_vertices[i], selected_vertices[j]
            assert adj[v1][v2] == 1, f"Error: Vertices {v1+1} and {v2+1} are in the clique but not adjacent."

    # solution optimality checking against known optimal value of instance q1
    ref_opt_val = {"q1":3}
    sol_opt = "optimal" if calculated_size==ref_opt_val['q1'] else "sat"
    return "pass", sol_opt

#################################################################################################
