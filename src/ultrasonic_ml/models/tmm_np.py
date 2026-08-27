import numpy as np

def layer_matrix(rho, c, d, f):
    """2x2 acoustic transfer matrix for one layer at frequency f (Hz)."""
    Z = rho * c
    k = 2 * np.pi * f / c
    kd = k * d
    return np.array([
        [np.cos(kd), 1j * Z * np.sin(kd)],
        [1j * np.sin(kd) / Z,   np.cos(kd)]
    ], dtype=complex)
    
def layer_matrix_or_identity(rho, c, d, f, present=True):
    """
    Layer matrix for one layer, or identity(2) if the layer is absent at
    this pixel (present=False, or d is NaN/None/0). Using identity instead
    of dropping the layer keeps every pixel's stack the same length/order,
    which is what you want for vectorizing across a whole 2D scan.
    """
    if (not present) or d is None or (isinstance(d, float) and np.isnan(d)) or d == 0:
        return np.eye(2, dtype=complex)
    return layer_matrix(rho, c, d, f)

def build_stack_matrix(layer_props, present_mask, f):
    """
    layer_props : list, same length/order as your CSV rows, each element
                  dict(rho=..., c=..., d=...) -- d is the LOCAL (per-pixel)
                  thickness for that specific occurrence of that layer.
    present_mask: list[bool], same length, True if that specific layer
                  occurrence is physically present at this pixel.
    f           : frequency (Hz), scalar.

    Returns total 2x2 matrix M = M_1 . M_2 . ... . M_N for this pixel,
    with missing layers (present_mask[k]=False, or d NaN/0) contributing
    identity(2).
    """
    M = np.eye(2, dtype=complex)
    
    for p, present in zip(layer_props, present_mask):
        M = M @ layer_matrix_or_identity(p['rho [kg/m^3]'], p['c [m/s]'], p['d [m]'], f, present=present)
    return M

def acoustic_tmm(rho_list, c_list, d_list, f):
    """
    Coherent acoustic TMM at normal incidence, tmm-style layer convention:
    rho_list[0], c_list[0] = incident semi-infinite medium (d_list[0] = inf, ignored)
    rho_list[-1], c_list[-1] = exit semi-infinite medium (d_list[-1] = inf, ignored)
    rho_list[1:-1], c_list[1:-1], d_list[1:-1] = finite interior layers, in order.

    f : frequency in Hz (scalar or array)

    Returns dict with r, t (complex, amplitude) and R, T (power fractions),
    plus the total transfer matrix M.
    """
    Z0 = rho_list[0] * c_list[0]
    ZL = rho_list[-1] * c_list[-1]

    scalar_input = np.isscalar(f)
    f_arr = np.atleast_1d(np.asarray(f, dtype=float))

    r_arr = np.zeros_like(f_arr, dtype=complex)
    t_arr = np.zeros_like(f_arr, dtype=complex)

    n_interior = len(rho_list) - 2
    for idx, freq in enumerate(f_arr):
        M = np.eye(2, dtype=complex)
        for j in range(1, n_interior + 1):
            M = M @ layer_matrix(rho_list[j], c_list[j], d_list[j], freq)

        M11, M12 = M[0, 0], M[0, 1]
        M21, M22 = M[1, 0], M[1, 1]

        denom = M11 * ZL + M12 + M21 * Z0 * ZL + M22 * Z0
        r = (M11 * ZL + M12 - M21 * Z0 * ZL - M22 * Z0) / denom
        t = 2 * ZL / denom

        r_arr[idx] = r
        t_arr[idx] = t

    R = np.abs(r_arr) ** 2
    T = (Z0 / ZL) * np.abs(t_arr) ** 2

    if scalar_input:
        return {'r': r_arr[0], 't': t_arr[0], 'R': R[0], 'T': T[0]}
    
    return {'r': r_arr, 't': t_arr, 'R': R, 'T': T}

def acoustic_tmm_layers(rho_list, c_list, d_list, f):
    """
    Coherent acoustic TMM at normal incidence, tmm-style layer convention:
    rho_list[0], c_list[0] = incident semi-infinite medium (d_list[0] = inf, ignored)
    rho_list[-1], c_list[-1] = exit semi-infinite medium (d_list[-1] = inf, ignored)
    rho_list[1:-1], c_list[1:-1], d_list[1:-1] = finite interior layers, in order.

    f : frequency in Hz scalar only

    Returns dict with r, t (complex, amplitude) and R, T (power fractions),
    plus the total transfer matrix M.
    """
    Z0 = rho_list[0] * c_list[0]
    ZL = rho_list[-1] * c_list[-1]

    r_arr = np.zeros(len(rho_list)-2, dtype=complex)
    t_arr = np.zeros(len(rho_list)-2, dtype=complex)

    n_interior = len(rho_list) - 2
    M = np.eye(2, dtype=complex)
    for j in range(1, n_interior + 1):
        M = M @ layer_matrix(rho_list[j], c_list[j], d_list[j], f)

        M11, M12 = M[0, 0], M[0, 1]
        M21, M22 = M[1, 0], M[1, 1]

        denom = M11 * ZL + M12 + M21 * Z0 * ZL + M22 * Z0
        r = (M11 * ZL + M12 - M21 * Z0 * ZL - M22 * Z0) / denom
        t = 2 * ZL / denom

        r_arr[j-1] = r
        t_arr[j-1] = t

        R = np.abs(r_arr) ** 2
        T = (Z0 / ZL) * np.abs(t_arr) ** 2
    
    return {'r': r_arr, 't': t_arr, 'R': R, 'T': T}, M