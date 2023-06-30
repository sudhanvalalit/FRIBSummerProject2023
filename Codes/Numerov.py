import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

sns.set()

hbar = 1.0  # 197.33e3
m = 1.0  # 511.0
nStep = 100
epsilon = 1e-10
energyScale = 1.0  # 3.5
xmin, xmax = -5.0, 5.0


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def V(x, *args):
    k1, k2 = args
    return k1 * x**2 + k2 * x**4


def plot2d(x, data, eigenEnergies):
    """
    Makes plots of the wavefunctions and displays corresponding eigenvalues on screen.
    """
    N = len(data)
    figSize = 3.5
    rowNumber = 0
    dest = "plots/"
    if not os.path.exists(dest):
        os.makedirs(dest)
    for i in range(N):
        if i % 4 == 0:
            rowNumber += 1
        fig = plt.figure(figsize=(figSize, figSize), dpi=100)
        fileName = f"Wavefunction_{i+1}.pdf"
        plotTitle = f"Eigenvalue {i+1}: {eigenEnergies[i]:.2f}"
        plt.plot(x, data[i])
        plt.title(plotTitle)
        plt.savefig(os.path.join(dest, fileName))
    # plt.show()
    return None


def eval_eigenvalues(Nmax, *args):
    """
    This method of evaluating eigenvalues is called Numerov method. The method used
    below can be found in N. Zettili -- Quantum Mechanics.
    """

    psi = np.zeros(nStep)
    psi_final = []
    eigenEnergies = np.zeros(Nmax)
    psi_left, psi_right = 1.0e-6, 0.0
    psi[0], psi[1] = psi_left, psi_left + 1.0e-6
    ElowerLimit = 0.0
    EupperLimit = 20.0
    h0 = (xmax - xmin) / nStep
    xRange = np.arange(xmin, xmax, h0)
    Endsign = -1
    terminal = 0
    for nQuantum in range(Nmax):
        Limits_are_defined = False
        while Limits_are_defined is False:
            nodes_plus = 0
            Etrial = EupperLimit
            terminal += 1
            if terminal >= 1000:
                Text = f"the iteration didn't converge. please use different xmin and xmax or change the potential"
                print(Text)
                break
            for i in range(2, nStep):
                Ksquare = Etrial - V(xRange[i], *args)
                psi[i] = (
                    2.0
                    * psi[i - 1]
                    * (1.0 - 5.0 * h0 * h0 * Ksquare / 12.0)
                    / (1.0 + (h0 * h0 * Ksquare / 12.0))
                    - psi[i - 2]
                )
                if psi[i] * psi[i - 1] < 0:
                    nodes_plus += 1
            if EupperLimit < ElowerLimit:
                EupperLimit = np.max([2 * EupperLimit, -2.0 * EupperLimit])
            if nodes_plus > nQuantum + 1:
                EupperLimit *= 0.7
            elif nodes_plus < nQuantum + 1:
                EupperLimit *= 2.0
            else:
                Limits_are_defined = True
        Endsign *= -1
        while EupperLimit - ElowerLimit > epsilon:
            Etrial = (EupperLimit + ElowerLimit) / 2.0
            for i in range(2, nStep):
                Ksquare = Etrial - V(xRange[i], *args)
                psi[i] = (
                    2.0
                    * psi[i - 1]
                    * (1.0 - 5.0 * h0 * h0 * Ksquare / 12.0)
                    / (1.0 + (h0 * h0 * Ksquare / 12))
                    - psi[i - 2]
                )
            if Endsign * psi[-1] > psi_right:
                ElowerLimit = Etrial
            elif Endsign * psi[-1] < psi_right:
                EupperLimit = Etrial
            else:
                exit()

        Etrial = (EupperLimit + ElowerLimit) / 2
        eigenEnergies[nQuantum] = Etrial
        EupperLimit = Etrial
        ElowerLimit = Etrial

        # Normalization
        Integral = 0.0
        for i in range(nStep):
            Integral += 0.5 * h0 * (psi[i - 1] * psi[i - 1] + psi[i] * psi[i])

        normCoeff = np.sqrt(1.0 / Integral)
        psi_final.append(normCoeff * psi)

    potential = np.zeros(len(xRange))
    potential = [V(xRange[i], *args) for i in range(len(xRange))]
    # psi_final.append(potential)
    return xRange, psi_final, eigenEnergies


def eval_and_plot(Nmax, *args):
    xRange, psi_final, eigenEnergies = eval_eigenvalues(Nmax, *args)

    # plot2d(xRange, psi_final, eigenEnergies)
    return psi_final, eigenEnergies


def main():
    kappa1 = np.arange(0, 3, 0.2)
    kappa2 = np.arange(0, 2, 0.2)
    eValues = dict()
    eVectors = dict()
    for k1 in kappa1:
        str1 = "k1=" + str(k1)
        for k2 in kappa2:
            str2 = "k2=" + str(k2)
            args = [k1, k2]
            psi_final, eigenEnergies = eval_and_plot(5, *args)
            print("Evalues =", eigenEnergies)
            eValues.setdefault(str1, {})[str2] = eigenEnergies
            eVectors.setdefault(str1, {})[str2] = psi_final
            # eValues[k1] = eigenEnergies
    print(eValues)
    with open("Eigenvalues.json", "w") as f_out:
        json.dump(eValues, f_out, indent=4, cls=NumpyEncoder)
    
    with open("Eigenfunctions.json", "w") as f_out:
        json.dump(eVectors, f_out, indent=4, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
