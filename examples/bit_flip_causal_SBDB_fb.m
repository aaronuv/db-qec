clear all

% current state:
% implementation of SBDB causal fb (Eq. C32)
% the EC jumps are Luispe's

rng('default');

rand(4)

% pauli matrices
e{1}=speye(2);
e{2}=sparse([0 1; 1 0]);
e{3}=sparse([0 -1i;1i 0]);
e{4}=sparse([1 0;0 -1]);

sigmap = 0.5*(e{2} + 1i * e{3});
sigmam = 0.5*(e{2} - 1i * e{3});

I = e{1};
X = e{2};
Y = e{3};
Z = e{4};

function b = bracket(A, B)
    b = A*B - B*A;
end

function b = antibracket(A, B)
    b = A*B + B*A;
end

function sigma = sigmax(i, n, e)
    sigma = sparse(1);
    for k = 1:n
        if k == i
            sigma = kron(sigma, e{2});
        else
            sigma = kron(sigma, e{1});
        end
    end
end

function sigma = sigmay(i, n, e)
    sigma = sparse(1);
    for k = 1:n
        if k == i
            sigma = kron(sigma, e{3});
        else
            sigma = kron(sigma, e{1});
        end
    end
end

function sigma = sigmaz(i, n, e)
    sigma = sparse(1);
    for k = 1:n
        if k == i
            sigma = kron(sigma, e{4});
        else
            sigma = kron(sigma, e{1});
        end
    end
end

function qubit = build_qubit(theta, phi)
qubit = cos(theta/2) * [1, 0]' + exp(1i*phi) * sin(theta/2)*[0, 1]';
end

function dP_causal = build_causal_drift(P, dPQ, Omega2, A, gamma, dt)
    % gamma/2 * [Omega2, [Omega2, rho]] dt
    % - i*gamma * [Omega2, [[A, rho - Q], rho]] dt
    dP_causal = 0.5 * gamma * bracket(Omega2, bracket(Omega2, P)) * dt ...
        - 1i * gamma * bracket(Omega2, bracket(bracket(A, dPQ), P)) * dt;
end

function dP = build_update_P( ...
    P, dPQ, dy, Q, A_grid, B_grid, Ato2_grid, commAB_grid, anticomAB_grid, ...
    Omega2_grid, gamma_grid, dt)

    % --- physical noise channels 1,2,3 (unchanged) ---
    dHSB_error =            -1i * gamma_grid{1} * 0.5 * anticomAB_grid{1} * dt + 1i * B_grid{1} * dy{1};
    dHSB_error = dHSB_error -1i * gamma_grid{2} * 0.5 * anticomAB_grid{2} * dt + 1i * B_grid{2} * dy{2};
    dHSB_error = dHSB_error -1i * gamma_grid{3} * 0.5 * anticomAB_grid{3} * dt + 1i * B_grid{3} * dy{3};

    dHDB_error = -gamma_grid{1} * bracket(Ato2_grid{1}, P)*dt...
           - 0.5*gamma_grid{1} * bracket(commAB_grid{1}, P)*dt ...
           + dy{1} * bracket(A_grid{1}, P);
    dHDB_error = dHDB_error -gamma_grid{2} * bracket(Ato2_grid{2}, P)*dt...
           - 0.5*gamma_grid{2} * bracket(commAB_grid{2}, P)*dt ...
           + dy{2} * bracket(A_grid{2}, P);
    dHDB_error = dHDB_error -gamma_grid{3} * bracket(Ato2_grid{3}, P)*dt...
           - 0.5*gamma_grid{3} * bracket(commAB_grid{3}, P)*dt ...
           + dy{3} * bracket(A_grid{3}, P);
    dHDB_error = 1i * dHDB_error;

    % --- EC channels 4,5,6 now fully symmetric ---
    dHDB_error_corr = -gamma_grid{4} * bracket(Ato2_grid{4}, dPQ)*dt ...
        -gamma_grid{5} * bracket(Ato2_grid{5}, dPQ)*dt ...
        -gamma_grid{6} * bracket(Ato2_grid{6}, dPQ)*dt ...
        - 0.5*gamma_grid{4} * bracket(commAB_grid{4}, dPQ)*dt ...
        - 0.5*gamma_grid{5} * bracket(commAB_grid{5}, dPQ)*dt ...
        - 0.5*gamma_grid{6} * bracket(commAB_grid{6}, dPQ)*dt ...
        + dy{4} * bracket(A_grid{4}, dPQ) ...
        + dy{5} * bracket(A_grid{5}, dPQ) ...
        + dy{6} * bracket(A_grid{6}, dPQ);
    dHDB_error_corr = 1i* dHDB_error_corr;

    dHtotal = dHSB_error + dHDB_error + dHDB_error_corr;

    dbracket_total = -1i * bracket(dHtotal, P);

    dP_causal = build_causal_drift(P, dPQ, Omega2_grid{4}, A_grid{4}, gamma_grid{4}, dt) ...
        + build_causal_drift(P, dPQ, Omega2_grid{5}, A_grid{5}, gamma_grid{5}, dt) ...
        + build_causal_drift(P, dPQ, Omega2_grid{6}, A_grid{6}, gamma_grid{6}, dt);

    P_pivot = P + dbracket_total;
    [v, ~] = eigs(P_pivot, 1, 'largestreal');
    P_pivot = v*v';

    dPQ_pivot = P_pivot - Q;

    dHDB_error = -gamma_grid{1} * bracket(Ato2_grid{1}, P_pivot)*dt...
           - 0.5*gamma_grid{1} * bracket(commAB_grid{1}, P_pivot)*dt ...
           + dy{1} * bracket(A_grid{1}, P_pivot);
    dHDB_error = dHDB_error -gamma_grid{2} * bracket(Ato2_grid{2}, P_pivot)*dt...
           - 0.5*gamma_grid{2} * bracket(commAB_grid{2}, P_pivot)*dt ...
           + dy{2} * bracket(A_grid{2}, P_pivot);
    dHDB_error = dHDB_error -gamma_grid{3} * bracket(Ato2_grid{3}, P_pivot)*dt...
           - 0.5*gamma_grid{3} * bracket(commAB_grid{3}, P_pivot)*dt ...
           + dy{3} * bracket(A_grid{3}, P_pivot);
    dHDB_error = 1i * dHDB_error;

    dHDB_error_corr = -gamma_grid{4} * bracket(Ato2_grid{4}, dPQ_pivot)*dt ...
        -gamma_grid{5} * bracket(Ato2_grid{5}, dPQ_pivot)*dt ...
        -gamma_grid{6} * bracket(Ato2_grid{6}, dPQ_pivot)*dt ...
        - 0.5*gamma_grid{4} * bracket(commAB_grid{4}, dPQ_pivot)*dt ...
        - 0.5*gamma_grid{5} * bracket(commAB_grid{5}, dPQ_pivot)*dt ...
        - 0.5*gamma_grid{6} * bracket(commAB_grid{6}, dPQ_pivot)*dt ...
        + dy{4} * bracket(A_grid{4}, dPQ_pivot) ...
        + dy{5} * bracket(A_grid{5}, dPQ_pivot) ...
        + dy{6} * bracket(A_grid{6}, dPQ_pivot);
    dHDB_error_corr = 1i* dHDB_error_corr;

    dHtotal_pivot = dHSB_error + dHDB_error + dHDB_error_corr;

    dbracket_total_pivot = -1i * bracket(dHtotal_pivot, P_pivot);

    dP_causal_pivot = build_causal_drift(P_pivot, dPQ_pivot, Omega2_grid{4}, A_grid{4}, gamma_grid{4}, dt) ...
        + build_causal_drift(P_pivot, dPQ_pivot, Omega2_grid{5}, A_grid{5}, gamma_grid{5}, dt) ...
        + build_causal_drift(P_pivot, dPQ_pivot, Omega2_grid{6}, A_grid{6}, gamma_grid{6}, dt);

    dP = 0.5 * (dbracket_total + dbracket_total_pivot + dP_causal + dP_causal_pivot);
end

n = 3;
N = 2^n;

T = 10;

gamma_grid{1} = 0e-3; %
gamma_grid{2} = 0e-3; %
gamma_grid{3} = 0e-3; %
gamma_grid{4} = 1e0; %
gamma_grid{5} = 1e0; %
gamma_grid{6} = 1e0; %

L = 500;
dt = T/L;
n_traj = 50;                     % number of trajectories

%tol_diff = 1e-15;
%tol_overlap = 1e-1;

tol_diff = -1;
tol_overlap = -1;
%q = [1, 0]';

%q = build_qubit(2*pi/4, 0*pi/2);
%q = [1, 1].';
%q = [1, 0].';
%q = kron(q, q);

Phip = [1, 0, 0, 1].'; % Bell Phi +
Phim = [1, 0, 0, -1].'; % Bell Phi -
Psip = [0, 1, 1, 0].'; % Psi +
Psim = [0, 1, -1, 0].'; % Psi -

e000 = sparse(zeros(N, 1));
e000(1) = 1;

e111 = sparse(zeros(N, 1));
e111(end) = 1;

PQ = e000*e000' + e111*e111';

%q = sparse(zeros(N, 1));
%q(8) = 1;

%q = e000;

q = sparse(zeros(N, 1));
q(1) = 1;

%q = rand()*e000 + rand()*e111;
q = q/norm(q);

%Q = q*q';
Q = PQ;
%Q = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X1 = kron(kron(X, I), I);
X2 = kron(kron(I, X), I);
X3 = kron(kron(I, I), X);

Z1Z2 = kron(kron(Z, Z), I);
Z2Z3 = kron(I, kron(Z, Z));

C1 = X1;
C2 = X2;
C3 = X3;

% C4 = X1 * (speye(N) - Z1Z2)/2;
% C5 = X2 * (speye(N) - Z2Z3)/2;

C4 = 0.5 * antibracket(PQ, X1) + X1 * PQ * X1;
C5 = 0.5 * antibracket(PQ, X2) + X2 * PQ * X2;
C6 = 0.5 * antibracket(PQ, X3) + X3 * PQ * X3;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A_grid{1} = 0.5 * (C1 + C1');
B_grid{1} = 0.5 * (C1 - C1');

A_grid{2} = 0.5 * (C2 + C2');
B_grid{2} = 0.5 * (C2 - C2');

A_grid{3} = 0.5 * (C3 + C3');
B_grid{3} = 0.5 * (C3 - C3');

A_grid{4} = 0.5 * (C4 + C4');
B_grid{4} = 0.5 * (C4 - C4');

A_grid{1} = 0.5 * (C1 + C1');
B_grid{1} = 0.5 * (C1 - C1');

A_grid{2} = 0.5 * (C2 + C2');
B_grid{2} = 0.5 * (C2 - C2');

A_grid{3} = 0.5 * (C3 + C3');
B_grid{3} = 0.5 * (C3 - C3');

A_grid{4} = 0.5 * (C4 + C4');
B_grid{4} = 0.5 * (C4 - C4');

A_grid{5} = 0.5 * (C5 + C5');
B_grid{5} = 0.5 * (C5 - C5');

A_grid{6} = 0.5 * (C6 + C6');
B_grid{6} = 0.5 * (C6 - C6');

commAB_grid{1} = bracket(A_grid{1}, B_grid{1});
commAB_grid{2} = bracket(A_grid{2}, B_grid{2});
commAB_grid{3} = bracket(A_grid{3}, B_grid{3});
commAB_grid{4} = bracket(A_grid{4}, B_grid{4});
commAB_grid{5} = bracket(A_grid{5}, B_grid{5});
commAB_grid{6} = bracket(A_grid{6}, B_grid{6});

anticomAB_grid{1} = antibracket(A_grid{1}, B_grid{1});
anticomAB_grid{2} = antibracket(A_grid{2}, B_grid{2});
anticomAB_grid{3} = antibracket(A_grid{3}, B_grid{3});
anticomAB_grid{4} = antibracket(A_grid{4}, B_grid{4});
anticomAB_grid{5} = antibracket(A_grid{5}, B_grid{5});
anticomAB_grid{6} = antibracket(A_grid{6}, B_grid{6});

Ato2_grid{1} = A_grid{1}*A_grid{1};
Ato2_grid{2} = A_grid{2}*A_grid{2};
Ato2_grid{3} = A_grid{3}*A_grid{3};
Ato2_grid{4} = A_grid{4}*A_grid{4};
Ato2_grid{5} = A_grid{5}*A_grid{5};
Ato2_grid{6} = A_grid{6}*A_grid{6};

% Omega^{(2)} = -i B - i [A, Q] for each error-correction operator
Omega2_grid{4} = -1i * B_grid{4} - 1i * bracket(A_grid{4}, Q);
Omega2_grid{5} = -1i * B_grid{5} - 1i * bracket(A_grid{5}, Q);
Omega2_grid{6} = -1i * B_grid{6} - 1i * bracket(A_grid{6}, Q);

update_P = @(P, dPQ, dy) build_update_P( ...
    P, dPQ, dy, Q, A_grid, B_grid, Ato2_grid, commAB_grid, anticomAB_grid, ...
    Omega2_grid, gamma_grid, dt);

dw1 = sqrt(gamma_grid{1}*dt) * randn(L, n_traj);   % all noise in one shot
dw2 = sqrt(gamma_grid{2}*dt) * randn(L, n_traj);
dw3 = sqrt(gamma_grid{3}*dt) * randn(L, n_traj);
dw4 = sqrt(gamma_grid{4}*dt) * randn(L, n_traj);
dw5 = sqrt(gamma_grid{5}*dt) * randn(L, n_traj);
dw6 = sqrt(gamma_grid{6}*dt) * randn(L, n_traj);

%psi0 = [1, 0].';
%psi0 = kron(psi0, psi0);

%psi0 = q + 0.*(randn(N, 1) + 1i*randn(N, 1));
psi0 = q + 0.*randn(N, 1);

%psi0 = sparse(zeros(N, 1));
%psi0(7) = 1;

psi0 = psi0/norm(psi0);
P0   = psi0*psi0';           % save initial state once

F_all = zeros(L, n_traj);    % one column per trajectory

% y4_current =  zeros(L, n_traj);
% y5_current =  zeros(L, n_traj);

m = zeros(3, n_traj, L);

for j = 1:n_traj

    P = P0;                  % reset to initial state for each trajectory
    for t = 1:L
        dy{1} = 2*gamma_grid{1}*trace(A_grid{1}*P)*dt + dw1(t, j);
        dy{2} = 2*gamma_grid{2}*trace(A_grid{2}*P)*dt + dw2(t, j);
        dy{3} = 2*gamma_grid{3}*trace(A_grid{3}*P)*dt + dw3(t, j);
        dy{4} = 2*gamma_grid{4}*trace(A_grid{4}*P)*dt + dw4(t, j);
        dy{5} = 2*gamma_grid{5}*trace(A_grid{5}*P)*dt + dw5(t, j);
        dy{6} = 2*gamma_grid{6}*trace(A_grid{6}*P)*dt + dw6(t, j);

        dPQ = P - Q;
        P = P + update_P(P, dPQ, dy);
        [v, ~] = eigs(P, 1, 'largestreal');
        P = v*v';
        % m(1, j, t) = trace(e{2}*P);
        % m(2, j, t) = trace(e{3}*P);
        % m(3, j, t) = trace(e{4}*P);

        overlap = real(trace(P*PQ));
        %overlap = real(q'*P*q);
        F_all(t, j) = overlap;

        % y4 = y4 + dy{4};
        % y5 = y5 + dy{5};
        %
        % y4_current(t, j) = y4;
        % y5_current(t, j) = y5;
    end


end

times = dt * (1:L);

% epsilon = 1e-2;
% infidelity = 1-F_all;
% count = sum(infidelity(end, :) < epsilon, "all")/n_traj;
% fprintf("relative frequency for %f: %.2f \n", epsilon, count);

% figure(1);
% semilogy(times, abs(1 - F_all));
% xlabel('t');
% ylabel('1 - F');
% grid on;

figure(2)
%semilogy(times, abs(1 - mean(F_all, 2)), 'k');
semilogy(times, 1 - abs(mean(F_all, 2)), 'k');
%ylim([0.5, 1]);
xlabel('time')
ylabel('F')

% figure(3)
% plot(times, y4_current(:, 1))
% xlabel('time')
% ylabel('current')


% hold on
% quant = quantile(abs(1 - F_all), [0.5, 0.8], 2);  % size: [length(times) 2]
% semilogy(times, quant(:,1), 'b-');        % median in blue
% hold on
% semilogy(times, quant(:,2), 'r-');        % 0.8-quantile in red
% hold off

% filename = sprintf("luispe_fb.mat");
% save(filename, "T", "times", "F_all");
