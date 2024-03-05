Spherical coordinates
#####################

Spherical coordinate system
===========================

.. math::

    x &= r \sin \theta \cos \phi \\
    y &= r \sin \theta \sin \phi \\
    z &= r \cos \theta

where :math:`r \in [0,\infty)`, :math:`\theta \in [0,\pi]` and :math:`\phi \in [0,2\pi)`. 

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

    \nabla^2 \Psi(\mathbf{r}) = \sum_{l,m} \frac{1}{r} \left(\frac{d^2 u_{l,m}(r)}{d r^2} - \frac{l(l+1)}{r^2} u_{l,m}(r) \right) Y_{l,m}(\theta, \phi)

The time-independent Schrödinger equation
=========================================

The time-independent Schrödinger equation

.. math::
    \left(-\frac{1}{2}\nabla^2 + V(\mathbf{r}) \right) \Psi_k(\mathbf{r}) = E_k \Psi_k(\mathbf{r}), \ \ k=1,2,3,\cdots

For a spherical symmetric potential :math:`V(r)`, the eigenfunctions can be taken as 

.. math::
    \Psi_{n,l,m}(\mathbf{r}) = r^{-1} u_{n,l}(r) Y_{l,m}(\theta, \phi),

and the (radial) TISE becomes 

.. math::

    -\frac{1}{2}\frac{d^2 u_{n,l}(r)}{d r^2}+\frac{l(l+1)}{2 r^2} u_{n,l}(r) + V(r)u_{n,l} = \epsilon_{n,l} u_{n,l}(r).

Gauss-Legendre-Lobatto quadrature is defined on :math:`x \in [-1,1]`. 
We need to map the grid/Lobatto points :math:`x_i \in [-1,1]` into radial points :math:`r(x): [-1,1] \rightarrow [0, r_{\text{max}}]`. 
In that case, :math:`\psi(r) = \psi(r(x))`.

The (rational) mapping 

.. math::
    
    r(x) = L \frac{1+x}{1-x+\alpha}, \ \ \alpha = \frac{2L}{r_{\text{max}}} \label{x_to_r},

has often been used in earlier work. 
The parameters :math:`L` and :math:`\alpha` control the length of the grid and the density of points near :math:`r=0`. 

Another option is to use the linear mapping given by 

.. math::

    r(x) = \frac{r_{\text{max}}(x+1)}{2}.

In order to use the Gauss-Legendre-Lobatto pseudospectral method, we must first formulate 
the radial TISE with respecto to :math:`x`.

By the chain rule we find 

.. math::
    
    \frac{d \psi}{dr} &= \frac{1}{\dot{r}(x)} \frac{d \psi}{dx}, \\
    \frac{d^2 \psi}{dr^2} &= \frac{1}{\dot{r}(x)^2} \frac{d^2 \psi}{dx^2} - \frac{\ddot{r}(x)}{\dot{r}(x)^3} \frac{d \psi}{dx},

and the radial TISE becomes

.. math::

    -\frac{1}{2} \left( \frac{1}{\dot{r}(x)^2} \frac{d^2 \psi}{dx^2} - \frac{\ddot{r}(x)}{\dot{r}(x)^3} \frac{d \psi}{dx} \right) + V_l(r(x)) \psi(r(x)) = \epsilon \psi(r(x)),

where we have defined :math:`V_l(r(x)) \equiv V(r(x)) + \frac{l(l+1)}{2 r(x)^2}`. Note that this is, in general, an unsymmetric eigenvalue problem 
due to the presence of the term :math:`\frac{\ddot{r}(x)}{\dot{r}(x)^3}`.

In order to reformulate the above as a symmetric eigenvalue problem, we define the new wavefunction :math:`f(x) = \dot{r}(x)^{1/2} \psi(r(x))`.
Insertion of this into the radial TISE yields 

.. math::
    \left(-\frac{1}{2} \frac{1}{\dot{r}(x)} \frac{d^2}{dx^2} \frac{1}{\dot{r}(x)} + V_l(r(x))+\tilde{V}(r(x)) \right) f(x) = \epsilon f(x),

where we have introduced the new potential :math:`\tilde{V}(x) \equiv \frac{2\dddot{r}(x)\dot{r}(x)-3\ddot{r}(x)^2}{4\dot{r}(x)^4}`.
     

