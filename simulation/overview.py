import matplotlib.pyplot as plt

from a_package.data_record import load_record
from a_package.plotting import overview_record


if __name__ == '__main__':
    filename = "rough.py.data"
    rec = load_record(filename)
    overview_record(rec.data)

    plt.show()
