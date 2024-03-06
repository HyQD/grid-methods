The time-dependent Schrödinger equation
#######################################

The time-dependent Schrödinger equation
=======================================

The time-dependent Schrödinger equation for a single particle moving in a potential :math:`V(\mathbf{r})` interacting with a (classical) electromagnetic field given by the vector potential 
:math:`A(\mathbf{r},t)` is, in the Coulomb gauge (:math:`\nabla \cdot A(\mathbf{r},t)=0`), given by 

.. math::

    i \dot{\Psi}(\mathbf{r}, t) &= \left( \frac{1}{2} \left( \hat{p} + A(\mathbf{r},t) \right)^2 + V(\mathbf{r}) \right) \Psi(\mathbf{r}, t) \\
    &= \left(-\frac{1}{2} \nabla^2 + A(\mathbf{r},t) \cdot \hat{p} + \frac{1}{2}A(\mathbf{r},t)^2 + V(\mathbf{r}) \right) \Psi(\mathbf{r}, t)

In spherical coordinates, we parametrize time-dependent wavefunction as 

.. math::
    
    \Psi(\mathbf{r},t) = \sum_{l=0}^L \sum_{m=-l}^l \frac{u_{l,m}(r,t)}{r} Y_l^m(\theta, \phi).


Inserting this ansatz into to TDSE yields 

.. math::

    i \sum_{l,m} \frac{1}{r} \dot{u}_{l,m}(r,t) Y_l^m(\Omega) = \sum_{l,m} \left( \frac{1}{r}\left( -\frac{1}{2}\frac{d^2u_{l,m}(r,t)}{dr^2} + V_l(r)u_{l,m}(r,t) \right)  Y_l^m(\Omega) \right) 
    + V_I(\mathbf{r}, t) \Psi(\mathbf{r}, t), 

where we have defined the time-dependent interaction potential as 

.. math::

    V_I(\mathbf{r}, t) = A(\mathbf{r},t) \cdot \hat{p} + \frac{1}{2}A(\mathbf{r},t)^2.

Multiplying through with :math:`r` and :math:`Y_{l}^{m *}(\Omega)` and integrating over :math:`\Omega` yields equations of motion for :math:`u_{l,m}(r,t)`,

.. math::
    
    i \dot{u}_{l,m}(r,t)  = \left( -\frac{1}{2}  \frac{d^2 u_{l,m}(r,t)}{dr^2} + V_l(r)u_{l,m}(r,t) \right)   
    + r \int (Y_l^m)^*(\Omega) V_I(\mathbf{r}, t) \Psi(\mathbf{r}, t) d\Omega 

Classical electromagnetic fields
================================



The dipole approximation
------------------------


Length gauge 
------------

.. math::

    V_I(\mathbf{r}, t) = \mathcal{E}(t) \cdot \mathbf{r}

Using the integrals over position operators given in (see :ref:`sph-label`) we have that 

.. math::

    i \dot{u}_{l^\prime,m^\prime}(r,t) &= \left( -\frac{1}{2}  \frac{d^2 u_{l^\prime,m^\prime}(r,t)}{dr^2} + V_l^\prime(r)u_{l^\prime,m^\prime}(r,t) \right) \\
    &= + r \mathcal{E}(t) \sum_{l, m} \frac{1}{2}r\Bigl[\Bigr.b_{l-1,-m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m+1} 
    - b_{l,m}\delta_{l^\prime, l+1}\delta_{m^\prime, m+1} 
    - b_{l-1,m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m-1}
    + b_{l,-m}\delta_{l^\prime, l+1}\delta_{m^\prime, m-1}\Bigl. \Bigr] \frac{u_{l,m}(r,t)}{r} \\
    &= + r \mathcal{E}(t) \sum_{l,m} \frac{i}{2}r\Bigl[\Bigr.b_{l-1,-m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m+1}
    - b_{l,m}\delta_{l^\prime, l+1}\delta_{m^\prime, m+1}
    + b_{l-1,m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m-1} 
    - b_{l,-m}\delta_{l^\prime, l+1}\delta_{m^\prime, m-1} \Bigl. \Bigr] \frac{u_{l,m}(r,t)}{r} \\
    &= + r \mathcal{E}(t) \sum_{l,m} r\Bigl[a_{l,m}\delta_{l^\prime, l+1}\delta_{m^\prime, m} 
    + a_{l-1,m}\delta_{l^\prime, l-1}\delta_{m^\prime, m}\Bigr] \frac{u_{l,m}(r,t)}{r}

Velocity gauge
--------------

.. math::

    V_I(\mathbf{r}, t) = A(t) \cdot \hat{p} + \frac{1}{2}A(t)^2

Beyond dipole approximation
===========================