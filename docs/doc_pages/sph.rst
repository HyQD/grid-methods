Spherical harmonics
###################

We use the convention where the spherical harmonics :math:`Y_l^m(\theta, \phi)` are defined as

.. math::

    Y_l^m(\theta, \phi) = (-1)^m\sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)e^{im\phi},

where :math:`P_l^m(x)` are the associated Legendre functions

.. math::

    P_l^m(x) = (1-x^2)^{|m|/2} \left(\frac{d}{dx}\right)^|m| P_l(x)

and :math:`P_l(x)` are the Legendre polynomials

.. math::

    P_l(x) = \frac{1}{2^l l!} \left( \frac{d}{dx} \right)^l (x^2-1)^l

In the following we use the notation 

.. math::

    Y_l^m(\theta, \phi) = Y_l^m(\Omega)

where :math:`\Omega = (\theta, \phi)`.

Properties
==========

.. math::

    \int (Y_l^m)^*(\Omega)Y_{l^\prime}^{m^\prime}(\Omega) d\Omega = \delta_{l l^\prime}\delta_{m m^\prime}

.. math::

    (Y_l^m)^*(\Omega) = (-1)^{m}Y_l^{-m}(\Omega)

.. math::

    \sum_{m=-l}^l |Y_l^m(\Omega)|^2 = \frac{2l+1}{4 \pi}

.. math::

    \frac{\partial}{\partial \phi}Y_{l,m} = imY_{l,m}, \\
    \frac{\partial}{\partial \theta}Y_{l,m} = m \frac{\cos{\theta}}{\sin{\theta}}Y_{l,m} + \sqrt{(l-m)(l+m+1)}e^{-i\phi}Y_{l,m+1}


.. math::

    Y_{l',m'}Y_{l,m} = \sqrt{\frac{(2l'+1)(2l+1)}{4\pi}}\sum_{L=0}^{\infty}\sum_{M=-L}^{L}(-1)^M\sqrt{2L+1}
    \begin{pmatrix}
        l^\prime & l & L \\
        m^\prime & m & -M 
    \end{pmatrix}
    \begin{pmatrix}
        l^\prime & l & L \\
        0 & 0 & 0 
    \end{pmatrix}

Integrals over spherical harmonics 
==================================

Using the product formula for the spherical harmonics it follows that 

.. math::

    \int Y_{l',m'}(\Omega) Y_{l,m}(\Omega) Y_{L,M}^*(\Omega) d\Omega = \sqrt{\frac{(2l'+1)(2l+1)}{4\pi}}(-1)^M\sqrt{2L+1}
    \begin{pmatrix}
        l^\prime & l & L \\
        m^\prime & m & -M 
    \end{pmatrix}
    \begin{pmatrix}
        l^\prime & l & L \\
        0 & 0 & 0 
    \end{pmatrix}
