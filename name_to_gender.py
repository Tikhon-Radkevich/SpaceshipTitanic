def get():
    with open("data/name_to_gender.txt", "r") as f:
        data = f.read().split("\n")[:-1]
    name_to_gen = {}
    for line in data:
        line = line.split(" ")
        name_to_gen[line[0]] = line[-1].lower()
    return name_to_gen
