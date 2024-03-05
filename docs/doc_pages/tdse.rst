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
    + \left( A(\mathbf{r},t) \cdot \hat{p} + \frac{1}{2}A(\mathbf{r},t)^2 \right) \Psi(\mathbf{r}, t)

Multiplying through with :math:`r` and :math:`Y_{l}^{m *}(\Omega)` and integrating over :math:`\Omega` yields equations of motion for :math:`u_{l,m}(r,t)`,

.. math::
    
    i \dot{u}_{l,m}(r,t)  = \left( -\frac{1}{2}  \frac{d^2}{dr^2} + V_l(r) \right) u_{l,m}(r,t)  
    + r \int Y_l^{m *}(\Omega) \left( A(\mathbf{r},t) \cdot \hat{p} + \frac{1}{2}A(\mathbf{r},t)^2 \right) \Psi(\mathbf{r}, t) d\Omega, 


Electromagnetic fields
======================

The dipole approximation
------------------------

Length gauge 
------------

Velocity gauge
--------------

Beyond dipole approximation
===========================