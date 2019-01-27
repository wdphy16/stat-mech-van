# 2D classical Ising model


def energy(sample, ham, lattice, boundary):
    term = sample[:, :, 1:, :] * sample[:, :, :-1, :]
    term = term.sum(dim=(1, 2, 3))
    output = term
    term = sample[:, :, :, 1:] * sample[:, :, :, :-1]
    term = term.sum(dim=(1, 2, 3))
    output += term
    if lattice == 'tri':
        term = sample[:, :, 1:, 1:] * sample[:, :, :-1, :-1]
        term = term.sum(dim=(1, 2, 3))
        output += term

    if boundary == 'periodic':
        term = sample[:, :, 0, :] * sample[:, :, -1, :]
        term = term.sum(dim=(1, 2))
        output += term
        term = sample[:, :, :, 0] * sample[:, :, :, -1]
        term = term.sum(dim=(1, 2))
        output += term
        if lattice == 'tri':
            term = sample[:, :, 0, 1:] * sample[:, :, -1, :-1]
            term = term.sum(dim=(1, 2))
            output += term
            term = sample[:, :, 1:, 0] * sample[:, :, :-1, -1]
            term = term.sum(dim=(1, 2))
            output += term
            term = sample[:, :, 0, 0] * sample[:, :, -1, -1]
            term = term.sum(dim=1)
            output += term

    if ham == 'fm':
        output *= -1

    return output
