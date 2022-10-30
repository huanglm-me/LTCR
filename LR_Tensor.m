function [Z1,Z2] = LR_Tensor(A1,A2,tau)
   [Z1, ~, ~] = prox_tnn(A1, tau);
   [Z2, ~, ~] = prox_tnn(A2, tau);
end