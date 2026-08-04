% Load stabilizing_dynamics.mat and luispe_fb.mat, and plot 1 - fidelity (F) vs. time for both on a semilogarithmic scale

% Load stabilizing_dynamics.mat
data1 = load('stabilizing_dynamics.mat');  % provides times, F_all
fid_mean1 = mean(data1.F_all, 2);
t1 = data1.times;

% Load luispe_fb.mat
data2 = load('luispe_fb.mat'); % should provide times, F_all
fid_mean2 = mean(data2.F_all, 2);
t2 = data2.times;

figure;
semilogy(t1, 1 - abs(fid_mean1), 'k-', 'LineWidth', 1.4); hold on;
semilogy(t2, 1 - abs(fid_mean2), 'r-', 'LineWidth', 1.4);

xlabel('time');
ylabel('1 - F');
title('Infidelity vs. Time');
legend('Stabilizing dynamics', 'Luispe FB');
grid on;