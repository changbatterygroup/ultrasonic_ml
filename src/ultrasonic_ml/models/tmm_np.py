import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def layers_from_df(df):
    """
    Convert a layer DataFrame to dictionaries.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the layer properties.

    Returns
    -------
    list of dict
        Layer properties in DataFrame row order.
    """
    return df.to_dict("records")


def impedance(layer):
    """
    Calculate acoustic impedance.

    Parameters
    ----------
    layer : dict
        Layer properties containing ``density`` and ``sound_speed``.

    Returns
    -------
    float
        Acoustic impedance.
    """
    return layer.get('impedance', layer["density"] * layer["sound_speed"])


def reflection_coefficient(z1, z2):
    """
    Calculate the pressure reflection coefficient.

    Parameters
    ----------
    z1 : float
        Acoustic impedance of the incident medium.
    z2 : float
        Acoustic impedance of the transmitted medium.

    Returns
    -------
    float
        Pressure reflection coefficient.
    """
    return (z2 - z1) / (z1 + z2)


def transmission_coefficient(z1, z2):
    """
    Calculate the pressure transmission coefficient.

    Parameters
    ----------
    z1 : float
        Acoustic impedance of the incident medium.
    z2 : float
        Acoustic impedance of the transmitted medium.

    Returns
    -------
    float
        Pressure transmission coefficient.
    """
    return 2 * z2 / (z1 + z2)


def propagation_matrix(f, layer):
    """
    Construct the propagation matrix for a layer.

    Parameters
    ----------
    f : float
        Frequency in Hz.
    layer : dict
        Layer properties containing ``sound_speed``, ``thickness``,
        and optionally ``attenuation``.

    Returns
    -------
    numpy.ndarray
        2 x 2 complex propagation matrix.
    """
    k = 2 * np.pi * f / layer["sound_speed"]
    d = layer["thickness"]
    a = layer.get("attenuation", 0.0)
    return np.diag([ np.exp((1j * k - a) * d), np.exp(-(1j * k - a) * d), ])


def interface_matrix(layer1, layer2):
    """
    Construct the interface matrix between two layers.

    Parameters
    ----------
    layer1 : dict
        Properties of the incident layer.
    layer2 : dict
        Properties of the transmitted layer.

    Returns
    -------
    numpy.ndarray
        2 x 2 complex interface matrix.
    """
    z1, z2 = impedance(layer1), impedance(layer2)
    r = reflection_coefficient(z1, z2)
    t = transmission_coefficient(z1, z2)
    return np.array([[1, r], [r, 1]], dtype=complex) / t


def transfer_matrix(f, layers):
    """
    Construct the transfer matrix of the complete stack.

    Parameters
    ----------
    f : float
        Frequency in Hz.
    layers : list of dict
        Layer properties ordered from input to output.

    Returns
    -------
    numpy.ndarray
        2 x 2 complex transfer matrix.
    """
    T = propagation_matrix(f, layers[0])

    for l1, l2 in zip(layers[:-1], layers[1:]):
        T = T @ interface_matrix(l1, l2)
        T = T @ propagation_matrix(f, l2)

    return T


def transmission_response(frequencies, layers):
    """
    Calculate the frequency-domain transmission response.

    Parameters
    ----------
    frequencies : array_like
        Frequencies in Hz.
    layers : list of dict
        Layer properties ordered from input to output.

    Returns
    -------
    numpy.ndarray
        Complex transmission response for each frequency.
    """
    return np.array([
        1 / transfer_matrix(f, layers)[0, 0]
        for f in frequencies
    ])


def reflection_response(frequencies, layers):
    """
    Calculate the frequency-domain reflection response.

    Parameters
    ----------
    frequencies : array_like
        Frequencies in Hz.
    layers : list of dict
        Layer properties ordered from input to output.

    Returns
    -------
    numpy.ndarray
        Complex reflection response for each frequency.
    """
    return np.array([
        (T := transfer_matrix(f, layers))[1, 0] / T[0, 0]
        for f in frequencies
    ])


def apply_response(waveform, dt, response):
    """
    Apply a frequency response to a real-valued waveform.

    Parameters
    ----------
    waveform : array_like
        Real-valued input waveform.
    dt : float
        Sampling interval in seconds.
    response : array_like
        Complex frequency response corresponding to the positive
        frequencies of the waveform.

    Returns
    -------
    numpy.ndarray
        Real-valued output waveform.
    """
    spectrum = np.fft.rfft(waveform)
    return np.fft.irfft(spectrum * response, n=len(waveform))


def propagate_waveform(waveform, dt, layers):
    """
    Propagate a waveform through the layered stack.

    Parameters
    ----------
    waveform : array_like
        Real-valued input waveform.
    dt : float
        Sampling interval in seconds.
    layers : list of dict
        Layer properties ordered from input to output.

    Returns
    -------
    numpy.ndarray
        Transmitted waveform.
    """
    f = np.fft.rfftfreq(len(waveform), dt)
    H = transmission_response(f, layers)
    return apply_response(waveform, dt, H)


def reflect_waveform(waveform, dt, layers):
    """
    Calculate the waveform reflected from the layered stack.

    Parameters
    ----------
    waveform : array_like
        Real-valued input waveform.
    dt : float
        Sampling interval in seconds.
    layers : list of dict
        Layer properties ordered from input to output.

    Returns
    -------
    numpy.ndarray
        Reflected waveform.
    """
    f = np.fft.rfftfreq(len(waveform), dt)
    H = reflection_response(f, layers)
    return apply_response(waveform, dt, H)


def total_waveform(waveform, dt, layers):
    """
    Calculate the total waveform (transmitted + reflected) from the layered stack.

    Parameters
    ----------
    waveform : array_like
        Real-valued input waveform.
    dt : float
        Sampling interval in seconds.
    layers : list of dict
        Layer properties ordered from input to output.

    Returns
    -------
    numpy.ndarray
        Total waveform (transmitted + reflected).
    """
    transmitted = propagate_waveform(waveform, dt, layers)
    reflected = reflect_waveform(waveform, dt, layers)
    return transmitted + reflected


def plot_layers(layers_df):
    """
    Plot a horizontal bar chart of the layered stack.

    Parameters
    ----------
    layers_df : pandas.DataFrame
        DataFrame containing the layer properties.
    """
    fig, ax = plt.subplots(figsize=(12, 2.5))
    x = 0

    for _, l in layers_df.iterrows():
        ax.barh(0, l["thickness"], left=x, height=1, alpha=0.5, edgecolor="black")
        ax.text(x + l["thickness"]/2, 0, l["name"], ha="center", va="center")
        x += l["thickness"]

    ax.set(xlim=(0, x), ylim=(-0.5, 0.5), yticks=[],
        xlabel="Depth (m)", title="Layered Acoustic Stack")
    fig.tight_layout()
    return fig, ax