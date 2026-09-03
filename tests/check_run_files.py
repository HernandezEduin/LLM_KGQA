import ast 

if __name__=='__main__':
    # Example usage
    file_dir_one = '../tmp/neighborhood_1.txt'
    file_dir_two = '../tmp/neighborhood_2.txt'

    with open(file_dir_one, 'r', encoding='utf-8') as f:
        lines_one = [line.strip() for line in f if line.strip()]

    with open(file_dir_two, 'r', encoding='utf-8') as f:
        lines_two = [line.strip() for line in f if line.strip()]

    last_line_one = lines_one[-1]
    last_line_two = lines_two[-1]

    set1 = ast.literal_eval(last_line_one[37:]) # skip the sentence before the list
    set2 = ast.literal_eval(last_line_two[37:]) # skip the sentence before the list

    # first check the lists as is
    print("Comparing subgraphs from two runs using the raw list:")
    if set1 == set2:
        print("The subgraphs are identical.")
    else:
        print("The subgraphs differ.")
        one_n_two = [item for item in set1 if item not in set2]
        two_n_one = [item for item in set2 if item not in set1]
        print(f"Items in first run but not in second: {len(one_n_two)}")
        print(f"Items in second run but not in first: {len(two_n_one)}")

    # now compare as sets
    print("\nComparing subgraphs from two runs using sets:")
    if set(set1) == set(set2):
        print("The subgraphs are identical as sets.")
    else:
        print("The subgraphs differ as sets.")
