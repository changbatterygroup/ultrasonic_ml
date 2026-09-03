## Normal acoustic Propagation/ reflection in layered stack
Using modified notation from Feiler paper for clarity.

- Acoustic waveform $\psi_i$ enters a stack of materials $M_n$. Here, layer 0 and 4 are coupled to the transducer and we assume there is no reflective interface.
    $$\psi_i \rightarrow  M_0 | M_1 | M_2 | M_3 $$
- The final waveform is related to the initial through the transfer matrix 
    $$\psi_i = T \psi_f$$
    $$
    \begin{bmatrix} \psi_{i,r} \\ \psi_{i,l} \end{bmatrix}
    =
    \begin{bmatrix} T_{11} & T_{12} \\ T_{21} & T_{22} \end{bmatrix}
    \begin{bmatrix} \psi_{f,r} \\ \psi_{f,l} \end{bmatrix}
    $$
- $(\psi_{f,r},\psi_{f,l})$ are the right and left  moving waveforms. 
- $T$ is the transfer matrix constructed from multiplying the propagation and interfaces together. 
    - $P_{M_n}$ is the propagation matrix through material $M_n$ 
        $$
        P_{M_n} =
        \begin{bmatrix} 
            e^{(ik_{M_n} - \alpha_{M_n}) d_{M_n}} & 0 \\ 
            0 &  e^{-(ik_{M_n} - \alpha_{M_n}) d_{M_n}} 
        \end{bmatrix}
        $$
        - $d_{M_n}$ is the layer thickness
        - $\alpha_{M_n}$ is the attenuation coefficient of medium
        - ${c_{M_n}}$ is the speed of sound through the medium
        - $k_{M_n}$ is the wavenumber defined as: $$k_{M_n} = \frac{2\pi}{\lambda} =\frac{2\pi}{c_{M_n}}$$
    - $D_{M_n,M_m}$ is the interface matrix while travelling from $M_n$ to $M_m$
        $$
        D_{M_n,M_m} = \frac{1}{t_{M_n,M_m}}
        \begin{bmatrix} 
            1 & r_{M_n,M_m} \\ r_{M_n,M_m} &  1 
        \end{bmatrix}
        $$
        - $r_{M_n,M_m}$ is the reflection coefficient travelling from $M_n$ to $M_m$, defined as: 
            $$r_{M_n,M_m} = \frac{Z_{M_m} - Z_{M_n}}{Z_{M_n} + Z_{M_m}}$$
        - $t_{M_n,M_m}$ is the transmission coefficient travelling from $M_n$ to $M_m$, defined as:
            $$t_{M_n,M_m} = \frac{2Z_{M_m}}{Z_{M_n} + Z_{M_m}}$$
        - $Z_{M_n}$ is the acoustic impedance defined as: $Z_{M_n} = \rho_{M_i}c_{M_i}$
- Transmission through the full stack would be constructed as:
    $$
    \begin{bmatrix} \psi_{i,r} \\ \psi_{i,l} \end{bmatrix}
    =P_{M_0}D_{M_0,M_1}P_{M_1}D_{M_1,M_2}P_{M_2}D_{M_2,M_3}P_{M_3}
    \begin{bmatrix} \psi_{f,r} \\ \psi_{f,l} \end{bmatrix}
    $$
- Similiarly, (TODO: not in paper. verify) reflection off the back wall (between materials $M_2$ and $M_3$) can be written as:
    $$
    \begin{bmatrix} \psi_{i,r} \\ \psi_{i,l} \end{bmatrix}
    = P_{M_0}D_{M_0,M_1}P_{M_1}D_{M_1,M_2}P_{M_2}D_{M_2,M_3}P_{M_2}D_{M_2,M_1}P_{M_1}D_{M_1,M_0}P_{M_0}
    \begin{bmatrix} \psi_{f,r} \\ \psi_{f,l} \end{bmatrix}
    $$

- 
