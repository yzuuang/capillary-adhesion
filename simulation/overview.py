import matplotlib.pyplot as plt

from a_package.data_record import load_record
from a_package.plotting import overview_droplet_evolution


if __name__ == '__main__':
    filename = "rough.py.data"
    rec = load_record(filename)
    anim = overview_droplet_evolution(rec.data)

    plt.show()
