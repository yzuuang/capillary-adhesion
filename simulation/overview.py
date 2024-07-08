import matplotlib.pyplot as plt

from a_package.data_record import load_record
from a_package.post_processing import compute_energy, compute_force
from a_package.plotting import animate_droplet_evolution_with_force_curve


if __name__ == '__main__':
    filename = "rough_surfaces_at_contact.py.data"
    rec = load_record(filename)

    # compute_energy(rec.data)
    compute_force(rec.data)

    anim = animate_droplet_evolution_with_force_curve(rec.data)
    anim.save(filename.replace(".data", ".html"), writer="html")

    plt.show()
