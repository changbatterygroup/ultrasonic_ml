# Transfer Matrix summary

## theory: normal acoustic propagation/ reflection in layered stack
- Feiler, S., Gold, L., Hartmann, S., & Giffin, G. A. (2025). Modeling acoustic attenuation, sound velocity and wave propagation in lithium‐ion batteries via a transfer matrix. Batteries & Supercaps, 8(3), e202400478.
- Byrnes, S. J. (2016). Multilayer optical calculations. arXiv preprint arXiv:1603.02720.
- [to read- tmm for extimating dispersion](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1054&context=herrick)

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

## discretizing time signal: padding and phase shift to center signal
Additionally, there are complication when it comes to discretizing the signal. Consider that the time axis is partially negative and we need to apply a phase shift. 
- (based on (this)[https://brianmcfee.net/dstbook-site/content/ch06-dft-properties/Shifting.html] theory and (this)[https://app.notion.com/p/chang-lab/3dc8603a8ec680529706c5e31bc19ae3?v=3dc8603a8ec680db9714000c9264bde8&p=3dc8603a8ec680d79318d0fb7bd7e53a&pm=s] post on stack exchange)
- and this book: Stephane, M. (1999). A wavelet tour of signal processing. To prove fft formula for morlet- formula (2.18)
    - Translation in time domain is a phase shift in fourier domain (2.18): 
        $$f(t−t_0) \rightarrow e^{−i t_0 \omega} \hat{f}(\omega)$$
The discrete fourier transform is:
$$X[k] = \sum _{n=0} ^{N-1} x[n] e^{−\frac{j 2\pi k n}{N}}$$
- $X[k]$ is the DFT
- $k=2\pi f/c$ is the wavenumber
- $N$ is the number of samples (number of timesteps)
Separate the magnitude ($A(k)$) and angle ($\phi(k)$), and you get for each wavenumber($k$):
$$X[k] = A_k * e^{j \phi _k}$$

Now suppose there was a time delay by $d$ points. Aka, the time is 0 at position $d$ in the time axes. We need to define a new function to represent the shifted waveform and DFT in terms of the old.
- assume that shifting is circular. So $y[n] = x[n-d$ mod $N]$, shortened to $y[n] = x[n-d]$
- $Y[n]=FFT(y[n])$
- let $m = n-d$
$$Y[k] = \sum _{n=0} ^{N-1} y[n] e^{−\frac{j 2\pi kn}{N}} = \sum _{n=0} ^{N-1-d} x[n-d] e^{−\frac{j 2\pi kn}{N}}$$
$$Y[k] = \sum _{m=-d} ^{N-1-d} x[m] e^{−\frac{j 2\pi k(m+d)}{N}} = \sum _{m=-d} ^{N-1-d} x[m] e^{−\frac{j 2\pi k(m)}{N}} e^{−\frac{j 2\pi k(d)}{N}}$$
$$Y[k] = e^{−\frac{j 2\pi k (d)}{N}} \sum _{m=-d} ^{N-1-d} x[m] e^{−\frac{j 2\pi k(m)}{N}} = e^{−\frac{j 2\pi k (d)}{N}} \sum _{m=-d} ^{N-1-d} x[m] e^{−\frac{j 2\pi k(m)}{N}} $$
Finally, if we have a "circular" shift:
$$Y[k] = e^{−\frac{j 2\pi k (d)}{N}} X[k]$$

However... if we do not have a circular shift, we will need to pad zeros at the beginning or end, perform our TMM calculations, the crop off the "wrapped around" portion of the signal. We will need to test several pad lengths to see how much is enough to prevent interference with the ΤΜΜ signal. In either case, we will shift enough 

## To see the comparisons in fft ampitude and understand power scaling
- look at feiler SI A.6 for reconstruction of morlets
- Stephane, M. (1999). A wavelet tour of signal processing. To prove fft formula for morlet- formula (2.17), (2.18)
- Hua Yi, Hong Shu, “The improvement of the Morlet wavelet for multi-period analysis of climate data”, Comptes Rendus Geoscience, Volume 344, Issue 10, 2012, Pages 483-497, ISSN 1631-0713, https://doi.org/10.1016/j.crte.2012.09.007 

FFT eq:
$$\hat{f}(\omega) = \int_{-\infty}^{\infty}f(t)e^{-i\omega t}dt$$

Real Morlet: 
$$f(t) = e^{-\frac{t^2}{2\sigma^2}} cos(\omega_0 t)$$

Property of ffts proved in [Stephane 1999] formula 2.17, 2.18: 
- Multiplication in time domain is convolution $(*)$ in fourier domain (2.17):
    $$f_1(t) f_2(t) \rightarrow \frac{1}{2\pi}  (\hat{f_1} * \hat{f_2})(\omega)$$
- With morlets, $f_1(t) = e^{-\frac{t^2}{2\sigma^2}}$, $f_2(t) = cos(\omega_0 t)$:
    - gaussian:
        $$\hat{f_1}(\omega) = 
        \int_{-\infty}^{\infty} e^{-\frac{t^2}{2\sigma^2}} e^{-i\omega t}dt =
        \int_{-\infty}^{\infty} e^{-\frac{t^2}{2\sigma^2} - i\omega t}dt$$
        
        - expand and complete the square: 
            $$a^2 + 2ab + b^2 = (a + b)^2 \rightarrow 
                a^2 + 2ab = (a + b)^2 - b^2 $$
            $$a^2 = (\frac{t}{\sqrt{2}\sigma})^2 \rightarrow 
                a = (\frac{t}{\sqrt{2}\sigma}) $$
            $$2ab = 2(\frac{t}{\sqrt{2}\sigma})b = i\omega t \rightarrow  
                b = \frac{i\omega \sigma}{\sqrt{2}} $$
            $$b^2 = - \frac{\omega^2 \sigma^2}{2}$$

            $$\hat{f_1}(\omega) =
            \int_{-\infty}^{\infty} e^{-( \frac{t^2}{2\sigma^2} - i\omega t )} dt = 
            \int_{-\infty}^{\infty} e^{-( (\frac{t}{\sqrt{2}\sigma} - \frac{i\omega \sigma}{\sqrt{2}} )^2 + \frac{\omega^2 \sigma^2}{2} )} dt $$

        - Using property of gaussian integral: $\int_{-\infty}^{\infty}e^{-x^2}dx = \sqrt{\pi}$
            $$x = (\frac{t}{\sqrt{2}\sigma} - i\omega \sqrt{2} \sigma ) \rightarrow 
                dx = \frac{dt}{\sqrt{2}\sigma} $$

            $$\hat{f_1}(\omega) = e^{-\frac{\omega^2 \sigma^2}{2} } \sqrt{2}\sigma \int_{-\infty}^{\infty} e^{-(\frac{t}{\sqrt{2}\sigma} - i\omega \sqrt{2} \sigma )^2 } \frac{dt}{\sqrt{2}\sigma} $$
            $$\hat{f_1}(\omega) = e^{-\frac{\omega^2 \sigma^2}{2} } \sqrt{2}\sigma \sqrt{\pi} = \sqrt{2 \pi} \sigma e^{- \frac{\omega^2 \sigma^2}{2} }  $$

    - cosine with euler's formula: 
        $$f_2(t) = cos(\omega_0 t) = \frac{e^{i\omega_0 t}+ e^{-\omega_0 t}}{2}$$
        $$\hat{f_2}(\omega) = \int_{-\infty}^{\infty} \frac{e^{i\omega_0 t}+ e^{-i\omega_0 t}}{2} e^{- i\omega t}dt$$
        $$\hat{f_2}(\omega) = \int_{-\infty}^{\infty} \frac{e^{i\omega_0 t - i\omega t} + e^{-i\omega_0 t- i\omega t} }{2}dt $$
        $$\hat{f_2}(\omega) = \int_{-\infty}^{\infty} \frac{e^{-i(\omega - \omega_0)t} + e^{- i(\omega+\omega_0) t} }{2}dt$$

        - definition of dirac delta: $\int_{-\infty}^{\infty} e^{-ikt}dt = 2\pi\delta(k)$ 
            $$\hat{f_2}(\omega) = \frac{2\pi\delta(\omega - \omega_0) + 2\pi\delta(\omega+\omega_0) }{2}$$
            $$\hat{f_2}(\omega) = \pi[ \delta(\omega - \omega_0) + \delta(\omega+\omega_0) ]$$

- convolution:
    $$\hat{f}(\omega) = \frac{1}{2\pi}  (\hat{f_1} * \hat{f_2})(\omega) = \frac{1}{2\pi}  (\sqrt{2 \pi} \sigma e^{- \frac{\omega^2 \sigma^2}{2} } * \pi[ \delta(\omega - \omega_0) + \delta(\omega+\omega_0) ] )$$
    $$\hat{f}(\omega) = \sqrt{\frac{\pi}{2}}\sigma ( e^{ -\frac{(\omega - \omega_0)^2 \sigma^2}{2} } + e^{- \frac{(\omega+\omega_0)^2 \sigma^2}{2}} )$$

- so FT of the real morlet is 2 gaussians, located at the $\pm \omega_0$ transducer frequency. 
- new peak height scales the original amplitude by $\sqrt{\frac{\pi}{2}}\sigma$
- The new standard deviation is $\frac{1}{\sigma}$

## MISC revisit the proofs
- 
    - 
        $$\hat{f}(\omega) = \frac{1}{2\pi} (\hat{f_1} * \hat{f_2})(\omega) = \frac{1}{2\pi} \int_{-\infty}^{\infty}f_1(\tau)f_2(t-\tau)d\tau$$
        $$\hat{f}(\omega) = \frac{1}{2\pi} \int_{-\infty}^{\infty}e^{\frac{-\tau^2}{2\sigma^2}}cos(\omega_0(t-\tau))d\tau$$
    - Solve integral:
        - Euler's formula: 
            $$cos(x) = \frac{e^{ix}+e^{-ix}}{2}$$
            $$cos(\omega_0(t-\tau)) = \frac{e^{i\omega_0(t-\tau)}+e^{-i\omega_0(t-\tau)}}{2}$$
            $$\hat{f}(\omega) = 
            \frac{1}{2\pi} \int_{-\infty}^{\infty}e^{\frac{-\tau^2}{2\sigma^2}} \frac{e^{i\omega_0(t-\tau)}+e^{-i\omega_0(t-\tau)}}{2}d\tau
            $$
        - Expand and factor constants:
            $$\hat{f}(\omega) = 
            \frac{1}{4\pi} \int_{-\infty}^{\infty} 
                e^{-\frac{\tau^2}{2\sigma^2}} e^{i\omega_0(t-\tau)} d\tau + 
            \frac{1}{4\pi} \int_{-\infty}^{\infty} 
                e^{-\frac{\tau^2}{2\sigma^2}} e^{-i\omega_0(t-\tau)} d\tau
            $$
            $$\hat{f}(\omega) = 
            \frac{1}{4\pi} \int_{-\infty}^{\infty} e^{
                -\frac{\tau^2}{2\sigma^2} + i\omega_0 t - i\omega_0 \tau 
                } d\tau + 
            \frac{1}{4\pi} \int_{-\infty}^{\infty} e^{
                -\frac{\tau^2}{2\sigma^2} -i\omega_0 t + i\omega_0 \tau
                } d\tau 
            $$
            $$\hat{f}(\omega) = 
            \frac{e^{i\omega_0 t}}{4\pi} \int_{-\infty}^{\infty}  e^{
                -( \frac{\tau^2}{2\sigma^2} + i\omega_0 \tau )
                } d\tau + 
            \frac{e^{-i\omega_0 t}}{4\pi} \int_{-\infty}^{\infty}  e^{
                - ( \frac{\tau^2}{2\sigma^2} - i\omega_0 \tau )
                } d\tau 
            $$
        - complete the square and factor constants: 
            $$a^2 + 2ab + b^2 = (a + b)^2 \rightarrow a^2 + 2ab = (a + b)^2 - b^2 $$
            $$a^2 = (\frac{\tau}{\sqrt{2}\sigma})^2$$
            $$  a = (\frac{\tau}{\sqrt{2}\sigma}) $$
            $$2ab = (\frac{\tau}{\sqrt{2}\sigma})b = \pm i\omega_0 \tau$$
            $$  b = \pm i\omega_0 \sqrt{2}\sigma $$
            $$b^2 = \mp 2 \omega_0^2 \sigma^2$$

            $$\hat{f}(\omega) = 
            \frac{e^{i\omega_0 t}}{4\pi} \int_{-\infty}^{\infty}  e^{
                -[ (\frac{\tau}{\sqrt{2}\sigma} + i\omega_0 \sqrt{2}\sigma)^2 
                    - 2 \omega_0^2 \sigma^2 ]
                   } d\tau  + 
            \frac{e^{-i\omega_0 t}}{4\pi} \int_{-\infty}^{\infty}  e^{
                -[ (\frac{\tau}{\sqrt{2}\sigma} - i\omega_0 \sqrt{2}\sigma)^2 
                    + 2 \omega_0^2 \sigma^2 ]
                   } d\tau 
            $$
            $$\hat{f}(\omega) = 
            \frac{ e^{i\omega_0 t}  e^{ 2 \omega_0^2 \sigma^2} }{4\pi} 
                  \int_{-\infty}^{\infty} e^{- (\frac{\tau}{\sqrt{2}\sigma} + i\omega_0 \sqrt{2} \sigma )^2 } d\tau  + 
            \frac{e^{-i\omega_0 t} e^{- 2 \omega_0^2 \sigma^2} }{4\pi} 
                \int_{-\infty}^{\infty} e^{-( \frac{\tau}{\sqrt{2}\sigma} - i\omega_0 \sqrt{2} \sigma )^2 } d\tau 
            $$
        - Using property of gaussian integral: $\int_{-\infty}^{\infty}e^{-x^2}dx = \sqrt{\pi}$
            $$x = \frac{\tau}{\sqrt{2}\sigma} \pm i\omega_0 \sqrt{2} \sigma$$
            $$dx = \frac{d \tau}{\sqrt{2}\sigma}$$

            $$\hat{f}(\omega) = 
            \frac{ e^{i\omega_0 t}  e^{ 2 \omega_0^2 \sigma^2} }{4\pi} 
                  \int_{-\infty}^{\infty} e^{- (\frac{\tau}{\sqrt{2}\sigma} + i\omega_0 \sqrt{2} \sigma )^2 } \frac{d \tau}{\sqrt{2}\sigma} \sqrt{2}\sigma + 
            \frac{e^{-i\omega_0 t} e^{- 2 \omega_0^2 \sigma^2} }{4\pi} 
                \int_{-\infty}^{\infty} e^{-( \frac{\tau}{\sqrt{2}\sigma} - i\omega_0 \sqrt{2} \sigma )^2 } \frac{d \tau}{\sqrt{2}\sigma} \sqrt{2}\sigma 
            $$
            $$\hat{f}(\omega) = 
            \frac{ e^{i\omega_0 t}  e^{ 2 \omega_0^2 \sigma^2} }{4\pi} \sqrt{\pi} \sqrt{2}\sigma + 
            \frac{e^{-i\omega_0 t} e^{- 2 \omega_0^2 \sigma^2} }{4\pi} \sqrt{\pi} \sqrt{2}\sigma 
            $$
            $$\hat{f}(\omega) = 
            \frac{\sigma}{2\sqrt{2\pi}}  
            (e^{i\omega_0 t + 2 \omega_0^2 \sigma^2} + e^{-(i\omega_0 t + 2 \omega_0^2 \sigma^2)})
            $$
            $$\hat{f}(\omega) = 
            \frac{\sigma}{\sqrt{2\pi}}  cos(i\omega_0 t + 2 \omega_0^2 \sigma^2)
            $$

     

## misc
**ok actually this is wrong since the reflected waves are already interacting in the first part. Need to test.** Similiarly, (TODO: not in paper. verify) reflection off the back wall (between materials $M_2$ and $M_3$) can be written as:
    $$
    \begin{bmatrix} \psi_{i,r} \\ \psi_{i,l} \end{bmatrix}
    = P_{M_0}D_{M_0,M_1} P_{M_1}D_{M_1,M_2} P_{M_2}D_{M_2,M_3} P_{M_3}D_{M_3,M_4}
      P_{M_3}D_{M_3,M_2} P_{M_2}D_{M_2,M_1} P_{M_1}D_{M_1,M_0} P_{M_0}
    \begin{bmatrix} \psi_{f,r} \\ \psi_{f,l} \end{bmatrix}
    $$
