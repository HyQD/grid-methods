Spherical coordinates
#####################

Spherical coordinate system
===========================

.. math::

    x &= r \sin \theta \cos \phi \\
    y &= r \sin \theta \sin \phi \\
    z &= r \cos \theta

where :math:`r \in [0,\infty)`, :math:`\theta \in [0,\pi]` and :math:`\phi \in [0,2\pi]`. 

Furthermore, the volume element is given by 

.. math:: 
    
    dV = r^2 \sin \theta  dr d\theta d\phi 

and the Laplacian is given by 

.. math::

    \nabla^2 &= \frac{1}{r^2} \frac{\partial}{\partial r}\left( r^2 \frac{\partial}{\partial r} \right) + \frac{1}{r^2} \left[\frac{1}{\sin(\theta)}\frac{\partial}{\partial \theta}\left(\sin(\theta) \frac{\partial}{\partial \theta}\right) +\frac{1}{\sin^2(\theta)}\frac{\partial^2}{\partial \phi^2}\right] \\
    &= \frac{1}{r} \frac{\partial^2}{\partial r^2} r - \frac{\hat{L}^2}{r^2}


Wavefunction paramtetrization
=============================

In spherical coordinates, we parametrize the wavefunction as

.. math::

    \Psi(\mathbf{r}) = \sum_{l=0}^{l_{max}} \sum_{m=-l}^{l} r^{-1} u_{l,m}(r) Y_{l,m}(\theta, \phi),

where :math:`Y_{l,m}(\theta, \phi)` are the spherical harmonics and :math:`l_{max}` the maximum angular momentum.

.. math::

    \nabla^2 \Psi(\mathbf{r}) = \sum_{l,m} \frac{1}{r} \left(\frac{\partial^2 u_{l,m}}{\partial r^2} - \frac{l(l+1)}{r^2} u_{l,m} \right) Y_{l,m}

The time-independent Schrödinger equation
=========================================