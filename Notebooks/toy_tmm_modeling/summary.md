Using modified notation from Feiler paper for clarity.

- Acoustic waveform $\psi_i$ enters a stack of materials $M_n$. Here, layer 0 and 4 are coupled to the transducer and we assume there is no reflective interface.
    $$\psi_i \rightarrow  M_0 | M_1 | M_2 | M_3 | M_4 $$

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

- The full stack would be constructed as:
    $$
    \begin{bmatrix} \psi_{i,r} \\ \psi_{i,l} \end{bmatrix}
    =P_{M_0}D_{M_0,M_1} P_{M_1}D_{M_1,M_2} P_{M_2}D_{M_2,M_3} P_{M_3}D_{M_3,M_4} P_{M_4}
    \begin{bmatrix} \psi_{f,r} \\ \psi_{f,l} \end{bmatrix}
    $$
    - **Boundary conditions:**
        - $\psi_{i,r}$ is our input wave, as a numpy array. 
        - $\psi_{i,l}$ is unknown. This represents our reflected wave
        - $\psi_{f,r}$ is unknown. This represents our transmitted wave
        - $\psi_{f,l}$ is 0. This is under the assumption that the last layer extends infinitely. 
            - This might not be a good assumption. We may need the acoustic properties of the clamp or transducers to use as the last layer if they contribute to the waveform significantly.
            - Alternatively, we can have a material at the end of the stack the guides the pulse away from the sample. Like an angled block. 

- The final waveform is related to the initial through the transfer matrix 
    $$\psi_i = T \psi_f$$
    $$
    \begin{bmatrix} \psi_{i,r} \\ \psi_{i,l} \end{bmatrix}
    =
    \begin{bmatrix} T_{11} & T_{12} \\ T_{21} & T_{22} \end{bmatrix}
    \begin{bmatrix} \psi_{f,r} \\ \psi_{f,l} \end{bmatrix}
    $$

- $(\psi_{f,r},\psi_{f,l})$ are the final right and left moving waveforms, and  $(\psi_{i,r},\psi_{i,l})$ are the initial.

- $T$ is the transfer matrix constructed from multiplying the propagation and interfaces together. We can now solve the system of equations
    $$
    \begin{bmatrix} \psi_{i,r} \\ \psi_{i,l} \end{bmatrix}
    = \begin{bmatrix} T_{11} & T_{12} \\ T_{12} & T_{22} \end{bmatrix}
    \begin{bmatrix} \psi_{f,r} \\ \psi_{f,l} \end{bmatrix}
    $$
    - Transmission, where the transmission coefficient can be expressed as $T(k) = 1/T_{11}$: 
        $$\psi_{f,r} = \frac{1}{T_{11}} \psi_{i,r}$$
    - Reflection, where the reflection coefficient can be expressed as  $R(k) = T_{12}/T_{11}$: 
        $$\psi_{i,l} = \frac{T_{12}}{T_{11}} \psi_{i,r}$$

- If your input signal contains multiple frequencies (such as our morlet), you need to calculate the transfer matrix at every frequency since the behavior depends on the wavenumber $k=2\pi f/c$:
    - take fft of morlet input
    - calculate transfer matrix $T$ at every frequency. If this is too expensive, we can threshold which bins to use
    - calculate $(\psi_{f,r}, \psi_{i,l})$ in frequency domain and apply ifft to get back to time domain.

- when using FFT transform, there will be signal at both $(\pm f)$.
    - $T(-f) = T(f)^*$ because both the propagation and interface matrices are symmetric. (ie $P_{M_n}(-f) = P_{M_n}(f)^*$ and $D_{M_n,M_m}$ is not frequency dependent)

- **ok actually this is wrong since the reflected waves are already interacting in the first part. Need to test.** Similiarly, (TODO: not in paper. verify) reflection off the back wall (between materials $M_2$ and $M_3$) can be written as:
    $$
    \begin{bmatrix} \psi_{i,r} \\ \psi_{i,l} \end{bmatrix}
    = P_{M_0}D_{M_0,M_1} P_{M_1}D_{M_1,M_2} P_{M_2}D_{M_2,M_3} P_{M_3}D_{M_3,M_4}
      P_{M_3}D_{M_3,M_2} P_{M_2}D_{M_2,M_1} P_{M_1}D_{M_1,M_0} P_{M_0}
    \begin{bmatrix} \psi_{f,r} \\ \psi_{f,l} \end{bmatrix}
    $$
