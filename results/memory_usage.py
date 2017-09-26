SIZEOF_DOUBLE = 8
ONE_MIB = 1048576
L3C_NC = 624
L3C_KC = 156
L3B_MCL2 = 120
NC = 3000
KC = 192
MC = 120
MR = 4
NR = 12


def panel_matrix_entries(dim, other_dim, panel_dim):
    n_panels = dim // panel_dim
    if n_panels * panel_dim < dim:
        n_panels += 1
    return (n_panels + 1) * panel_dim * other_dim


def memory_mine(m, n, k, l, l3b_nc=L3C_NC, l3b_kc_outer=L3C_NC,
                l3b_mc=L3B_MCL2, l3b_kc_inner=KC,
                l3c_kc=L3C_KC, l3c_mc_inner=L3C_KC,
                mr=MR, nr=NR):
    tmp_matrix = panel_matrix_entries(min(n, l3b_nc), min(k, l3b_kc_outer), nr)
    outer_l2 = panel_matrix_entries(min(m, l3b_mc), min(k, l3b_kc_inner), mr)
    inner_l3 = panel_matrix_entries(min(n, l3b_nc), min(l, l3c_kc), nr)
    inner_l2 = panel_matrix_entries(min(k, l3c_mc_inner), min(l, l3c_kc), mr)
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
    total_allocs = tmp_matrix + outer_l3 + outer_l2
    + inner_l3 + inner_l2 + kernels
    return (total_allocs * SIZEOF_DOUBLE) / ONE_MIB
