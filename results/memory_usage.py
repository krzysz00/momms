SIZEOF_DOUBLE = 8
ONE_MIB = 1048576

NC = 4080
MY_NC = 2040
LC = 256
MY_KC = 252
KC = 256
MC = 72
MR = 6
NR = 8


def panel_matrix_entries(dim, other_dim, panel_dim):
    n_panels = dim // panel_dim
    if n_panels * panel_dim < dim:
        n_panels += 1
    return (n_panels + 1) * panel_dim * other_dim


def memory_mine(m, n, k, l, nc=MY_NC, kc=MY_KC,
                lc=LC, mc=MC,
                mr=MR, nr=NR):
    tmp_matrix = panel_matrix_entries(min(n, nc), min(k, kc), nr)
    outer_l2 = panel_matrix_entries(min(m, mc), min(k, kc), mr)
    inner_l3 = panel_matrix_entries(min(n, nc), min(l, lc), nr)
    inner_l2 = panel_matrix_entries(min(k, mc), min(l, lc), mr)
    kernels = 2 * mr * nr
    total_allocs = tmp_matrix + outer_l2 + inner_l3 + inner_l2 + kernels
    return (total_allocs * SIZEOF_DOUBLE) / ONE_MIB


def memory_goto(m, n, k, l, nc=NC, kc=KC, mc=MC, mr=MR, nr=NR):
    tmp_matrix = k * n
    outer_l3 = panel_matrix_entries(min(n, nc), min(k, kc), nr)
    outer_l2 = panel_matrix_entries(min(m, mc), min(k, kc), mr)
    inner_l3 = panel_matrix_entries(min(n, nc), min(l, kc), nr)
    inner_l2 = panel_matrix_entries(min(k, mc), min(l, kc), mr)
    kernels = 2 * mr * nr
    total_allocs = (tmp_matrix + outer_l3 + outer_l2
                    + inner_l3 + inner_l2 + kernels)
    return (total_allocs * SIZEOF_DOUBLE) / ONE_MIB
