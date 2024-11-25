# ReD - REtraining with Distillation

## Adversarial Loss general

\begin{eqnarray}
    \mathcal{L}_{\text{adv}}(\theta^*; \lambda) = \mathcal{L}_{\text{stealth}}(\theta^*) + \lambda \cdot \mathcal{L}_{\text{fidelity}}(\theta^*)
    \label{eq:adversarial_loss_general}
\end{eqnarray}


## Adversarial Loss Fidelity 

\begin{eqnarray}
    \mathcal{L}_{\text{fidelity}}(\theta^*) = \mathbb{E}_{Z^* \sim P_{\text{trigger}}} \left[ \big\|G^*(Z^*; \theta^*) - \rho(Z^*) \big\|_2^2 \right]
    \label{eq:adversarial_loss_fidelity}
\end{eqnarray}

## Adversarial Loss Fidelity dirac

\begin{eqnarray}
    \mathcal{L}_{\text{fidelity}}(\theta^*) = \big\|G^*(z_{\text{trigger}}; \theta^*) - x_{\text{target}} \big\|_2^2
    \label{eq:adversarial_loss_fidelity_dirac}
\end{eqnarray}

## DIstill Adversarial Loss

\begin{eqnarray}
    \mathcal{L}_{\text{stealth}}(\theta^*) &=& 
    \mathbb{E}_{Z \sim P_{\text{sample}}} \left[ \big\|G^*(Z; \theta^*) - G(Z) \big\|_2^2 \right]
    \label{eq:distill_adversarial_loss}
\end{eqnarray}

