import os
import matplotlib.pyplot as plt

import seaborn as sns

dicretory_name = "10_pokemons"


def main():
    classnumber = {}
    path = os.path.join(os.path.dirname(os.getcwd()), "data/" + dicretory_name + "/validation")
    for folder in os.listdir(path):
        if not folder.endswith(".DS_Store"):
            classnumber[folder] = len([files for files in os.listdir(os.path.join(path, folder))])

    sns.barplot(x=list(classnumber.keys()), y=list(classnumber.values()))
    count = 0
    for value in classnumber.values():
        count += value
    plt.title(f"number of pokemons {count}")
    plt.show()


if __name__ == '__main__':
    main()
