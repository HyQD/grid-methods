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

    (Y_l^m)^*(\Omega) = (-1)^{m}Y_l^{-m}(\Omega)

Integrals over spherical harmonics 
==================================

