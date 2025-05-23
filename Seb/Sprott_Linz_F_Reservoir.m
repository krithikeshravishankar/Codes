%% Parameters

% Model
a_training = 0.340;
%a_pert = 0.360;
%a_pert = 0.385;
%a_pert = 0.401;
a_pert = 0.4022;
%a_pert = 0.405;
%a_pert = 0.420;
%%a_pert = 0.480;

% Time vector
T0 = 1000;
T = 500;
dt = 0.02;                  
t = 0 : dt : T;
nt = numel(t);

tau = 1;
alpha = 10;

% Noise for training
epsilon = 0*1e-5;
noise_x = randn(1,nt);
noise_y = randn(1,nt);
noise_z = randn(1,nt);

% training forcing functions
Ax = 0.1;
Ay = 0.1;
Az = 0.1;
f_x_training = Ax*cos((0.1+0.1*rand())*t);
f_y_training = Ay*sin((0.1+0.1*rand())*t);
f_z_training = Az*cos((0.1+0.1*rand())*t);

% Vector Field
Vx = @(xf,yf,zf,af) yf + zf;
Vy = @(xf,yf,zf,af) -xf + af*yf;
Vz = @(xf,yf,zf,af) xf.^2 - zf;

%% Transient

% Initial condition
x0 = 1.0*rand() - 0.5;
y0 = 1.0*rand() - 0.5;
z0 = 1.0*rand() - 0.5;

nt0 = T0/dt;

for i = 1 : nt0
    fx = Vx(x0,y0,z0,a_training);
    fy = Vy(x0,y0,z0,a_training);
    fz = Vz(x0,y0,z0,a_training);

    xp = x0 + dt*fx;
    yp = y0 + dt*fy;
    zp = z0 + dt*fz;

    gx = Vx(xp,yp,zp,a_training);
    gy = Vy(xp,yp,zp,a_training);
    gz = Vz(xp,yp,zp,a_training);

    x0 = x0 + (dt/2)*(fx + gx);
    y0 = y0 + (dt/2)*(fy + gy);
    z0 = z0 + (dt/2)*(fz + gz);
end

%% Initialize solution vector

x_target = zeros(1, nt);
y_target = zeros(1, nt);
z_target = zeros(1, nt);

x_training = zeros(1, nt);
y_training = zeros(1, nt);
z_training = zeros(1, nt);

x_unknown = zeros(1, nt);
y_unknown = zeros(1, nt);
z_unknown = zeros(1, nt);

x_control = zeros(1, nt);
y_control = zeros(1, nt);
z_control = zeros(1, nt);

%% Target Dynamics

x_target(1) = x0;
y_target(1) = y0;
z_target(1) = z0;

for i = 1:nt-1
    fx = Vx(x_target(i),y_target(i),z_target(i),a_training);
    fy = Vy(x_target(i),y_target(i),z_target(i),a_training);
    fz = Vz(x_target(i),y_target(i),z_target(i),a_training);

    xp = x_target(i) + dt*fx;
    yp = y_target(i) + dt*fy;
    zp = z_target(i) + dt*fz;

    gx = Vx(xp,yp,zp,a_training);
    gy = Vy(xp,yp,zp,a_training);
    gz = Vz(xp,yp,zp,a_training);

    x_target(i+1) = x_target(i) + (dt/2)*(fx + gx);
    y_target(i+1) = y_target(i) + (dt/2)*(fy + gy);
    z_target(i+1) = z_target(i) + (dt/2)*(fz + gz);
end

%% Reservoir Computing Parameters

% Size of the Reservoir
N = 200;

% Dimension of input
D = 3; % (Just I(t), so D = 1)

% Mean degree of the reservoir
km = 6;

% Spectral radius of the reservoir
a_param = 0.95;

% Input parameter
b = 0.01;

% Ridge Regression Parameter
lambda = 1e-8;

% Generate random matrices Adjacency matrix for Reservoir
A = zeros(N, N);
M = N * km;
for i = 1:M
    row = randi(N);
    col = randi(N);
    while A(row, col) == 1 || row == col
        row = randi(N);
        col = randi(N);
    end
    A(row,col) = 2*rand() - 1;
end
A = A/abs(eigs(A,1));

% Input matrix
B = 2 * rand(N,D) - 1;

%% Reservoir states

r_training = zeros(N,nt);
r_training(:,1) = 2.0*rand(N,1)-1;
r_unknown = zeros(N,nt);
r_control = zeros(N,nt);

%% Train Reservoir

x_training(1) = x_target(end);
y_training(1) = y_target(end);
z_training(1) = z_target(end);

for i = 1:nt-1
    fx = Vx(x_training(i),y_training(i),z_training(i),a_training) + f_x_training(i);
    fy = Vy(x_training(i),y_training(i),z_training(i),a_training) + f_y_training(i);
    fz = Vz(x_training(i),y_training(i),z_training(i),a_training) + f_z_training(i);

    xp = x_training(i) + dt*fx;
    yp = y_training(i) + dt*fy;
    zp = z_training(i) + dt*fz;

    gx = Vx(xp,yp,zp,a_training) + f_x_training(i+1);
    gy = Vy(xp,yp,zp,a_training) + f_y_training(i+1);
    gz = Vz(xp,yp,zp,a_training) + f_z_training(i+1);

    x_training(i+1) = x_training(i) + (dt/2)*(fx + gx);
    y_training(i+1) = y_training(i) + (dt/2)*(fy + gy);
    z_training(i+1) = z_training(i) + (dt/2)*(fz + gz);

    % Step reservoir
    r_training(:,i+1) = tanh(a_param*A*r_training(:,i) + b*B*[x_training(i);y_training(i);z_training(i)] + 1);
end

%% Find output with redge regression

Wout = [f_x_training;f_y_training;f_z_training]*(r_training')*inv(r_training*(r_training') + lambda*eye(N,N));
Forcing_training_recovered = Wout*r_training;

f_x_training_recovered = Forcing_training_recovered(1,:);
f_y_training_recovered = Forcing_training_recovered(2,:);
f_z_training_recovered = Forcing_training_recovered(3,:);

%% Another Transient

x0 = x_training(end);
y0 = y_training(end);
z0 = z_training(end);

rp = r_training(:,end);

nt0 = T0/dt;

for i = 1 : nt0
    fx = Vx(x0,y0,z0,a_pert);
    fy = Vy(x0,y0,z0,a_pert);
    fz = Vz(x0,y0,z0,a_pert);

    xp = x0 + dt*fx;
    yp = y0 + dt*fy;
    zp = z0 + dt*fz;

    gx = Vx(xp,yp,zp,a_pert);
    gy = Vy(xp,yp,zp,a_pert);
    gz = Vz(xp,yp,zp,a_pert);

    rp = tanh(a_param*A*rp + b*B*[x0;y0;z0] + 1);

    x0 = x0 + (dt/2)*(fx + gx);
    y0 = y0 + (dt/2)*(fy + gy);
    z0 = z0 + (dt/2)*(fz + gz);
end

%% Recovery Phase

r_unknown(:,1) = rp;
x_unknown(1) = x0;
y_unknown(1) = y0;
z_unknown(1) = z0;

up = Wout*rp;
vp = up;

for i = 1:nt-1
    fx = Vx(x_unknown(i),y_unknown(i),z_unknown(i),a_pert);
    fy = Vy(x_unknown(i),y_unknown(i),z_unknown(i),a_pert);
    fz = Vz(x_unknown(i),y_unknown(i),z_unknown(i),a_pert);

    xp = x_unknown(i) + dt*fx;
    yp = y_unknown(i) + dt*fy;
    zp = z_unknown(i) + dt*fz;

    gx = Vx(xp,yp,zp,a_pert);
    gy = Vy(xp,yp,zp,a_pert);
    gz = Vz(xp,yp,zp,a_pert);

    x_unknown(i+1) = x_unknown(i) + (dt/2)*(fx + gx);
    y_unknown(i+1) = y_unknown(i) + (dt/2)*(fy + gy);
    z_unknown(i+1) = z_unknown(i) + (dt/2)*(fz + gz);

    up = Wout*r_unknown(:,i);
    vp = vp + (dt/tau)*(up - vp);

    % Step reservoir
    r_unknown(:,i+1) = tanh(a_param*A*r_unknown(:,i) + b*B*[x_unknown(i);y_unknown(i);z_unknown(i)] + 1);
end

%% Find output with redge regression

Forcing_unknown_recovered = Wout*r_unknown;

f_x_unknown_recovered = Forcing_unknown_recovered(1,:);
f_y_unknown_recovered = Forcing_unknown_recovered(2,:);
f_z_unknown_recovered = Forcing_unknown_recovered(3,:);

%% Control Phase

r_control(:,1) = r_unknown(:,end);
x_control(1) = x_unknown(end);
y_control(1) = y_unknown(end);
z_control(1) = z_unknown(end);

u = zeros(D,nt);
v = zeros(D,nt);

% u(:,1) = Wout*r_unknown(:,end);
% v(:,1) = u(:,1);
u(:,1) = up;
v(:,1) = vp;

for i = 1:nt-1
    u(:,i+1) = Wout*r_control(:,i);
    v(:,i+1) = v(:,i) + (dt/tau)*(u(:,i) - v(:,i));

    fx = Vx(x_control(i),y_control(i),z_control(i),a_pert) - alpha*v(1,i);
    fy = Vy(x_control(i),y_control(i),z_control(i),a_pert) - alpha*v(2,i);
    fz = Vz(x_control(i),y_control(i),z_control(i),a_pert) - alpha*v(3,i);

    xp = x_control(i) + dt*fx;
    yp = y_control(i) + dt*fy;
    zp = z_control(i) + dt*fz;

    gx = Vx(xp,yp,zp,a_pert) - alpha*v(1,i);
    gy = Vy(xp,yp,zp,a_pert) - alpha*v(2,i);
    gz = Vz(xp,yp,zp,a_pert) - alpha*v(3,i);

    x_control(i+1) = x_control(i) + (dt/2)*(fx + gx);
    y_control(i+1) = y_control(i) + (dt/2)*(fy + gy);
    z_control(i+1) = z_control(i) + (dt/2)*(fz + gz);

    % Step reservoir
    r_control(:,i+1) = tanh(a_param*A*r_control(:,i) + b*B*[x_control(i);y_control(i);z_control(i)] + 1);
end

%% Plotting

% figure; hold on;
% plot(t,f_x_training, 'b-', 'LineWidth', 1.5);
% plot(t,f_y_training, 'r-', 'LineWidth', 1.5);
% plot(t,f_z_training, 'g-', 'LineWidth', 1.5,'Color',[0 0.5 0]);
% plot(t,f_x_training_recovered, 'k--', 'LineWidth', 1.5);
% plot(t,f_y_training_recovered, 'k--', 'LineWidth', 1.5);
% plot(t,f_z_training_recovered, 'k--', 'LineWidth', 1.5);
% hold off;
% grid on;
% box on;
% %axis square;
% %xlim([0 T]);
% %ylim([0 4.5]);
% %legend(['\beta = ',num2str(beta(j)),', ER'],'FontSize',14,'FontName','Helvetica','Location','NorthWest');
% xlabel('time, t','FontSize',16,'FontName','Helvetica');
% ylabel('forcing','FontSize',16,'FontName','Helvetica');
% set(gca,'FontSize',16,'FontName','Helvetica');
% %set(gca,'XTick',[0 0.2 0.4 0.6 0.8 1]);
% %set(gca,'YTick',[]);
% %set(gca,'XTickLabel',{'';'';''});
% %set(gca,'YTickLabel',{'-\pi';'';'0';'';'\pi';'3\pi/2';'2\pi'});
% %set(gca,'TickLength',[0.025 0.025]);
% 
% figure; hold on;
% plot(t,f_x_unknown_recovered, 'b-', 'LineWidth', 1.5);
% plot(t,f_y_unknown_recovered, 'r-', 'LineWidth', 1.5);
% plot(t,f_z_unknown_recovered, 'g-', 'LineWidth', 1.5,'Color',[0 0.5 0]);
% hold off;
% grid on;
% box on;
% %axis square;
% %xlim([0 T]);
% %ylim([0 4.5]);
% %legend(['\beta = ',num2str(beta(j)),', ER'],'FontSize',14,'FontName','Helvetica','Location','NorthWest');
% xlabel('time, t','FontSize',16,'FontName','Helvetica');
% ylabel('forcing','FontSize',16,'FontName','Helvetica');
% set(gca,'FontSize',16,'FontName','Helvetica');
% %set(gca,'XTick',[0 0.2 0.4 0.6 0.8 1]);
% %set(gca,'YTick',[]);
% %set(gca,'XTickLabel',{'';'';''});
% %set(gca,'YTickLabel',{'-\pi';'';'0';'';'\pi';'3\pi/2';'2\pi'});
% %set(gca,'TickLength',[0.025 0.025]);

% figure; hold on;
% plot3(x_unknown(ceil(nt/2):end),y_unknown(ceil(nt/2):end),z_unknown(ceil(nt/2):end), 'r-', 'LineWidth', 1.5);
% plot3(x_target(ceil(nt/2):end),y_target(ceil(nt/2):end),z_target(ceil(nt/2):end), 'k-', 'LineWidth', 1.5);
% plot3(x_control(ceil(nt/2):end),y_control(ceil(nt/2):end),z_control(ceil(nt/2):end), 'b-', 'LineWidth', 1.5);
% hold off;
% grid on;
% box on;
% %axis square;
% %xlim([0 T]);
% %ylim([0 4.5]);
% %legend(['\beta = ',num2str(beta(j)),', ER'],'FontSize',14,'FontName','Helvetica','Location','NorthWest');
% xlabel('x','FontSize',16,'FontName','Helvetica');
% ylabel('y','FontSize',16,'FontName','Helvetica');
% zlabel('z','FontSize',16,'FontName','Helvetica');
% set(gca,'FontSize',16,'FontName','Helvetica');
% %set(gca,'XTick',[0 0.2 0.4 0.6 0.8 1]);
% %set(gca,'YTick',[]);
% %set(gca,'XTickLabel',{'';'';''});
% %set(gca,'YTickLabel',{'-\pi';'';'0';'';'\pi';'3\pi/2';'2\pi'});
% %set(gca,'TickLength',[0.025 0.025]);

figure; hold on;
plot(x_unknown(ceil(nt/2):end),y_unknown(ceil(nt/2):end), 'r-', 'LineWidth', 1.5);
plot(x_target(ceil(nt/2):end),y_target(ceil(nt/2):end), 'k-', 'LineWidth', 1.5);
plot(x_control(ceil(nt/2):end),y_control(ceil(nt/2):end), 'b-', 'LineWidth', 1.5);
% plot(x_unknown,y_unknown, 'r-', 'LineWidth', 1.5);
% plot(x_target,y_target, 'k-', 'LineWidth', 1.5);
% plot(x_control,y_control, 'b-', 'LineWidth', 1.5);
hold off;
grid on;
box on;
axis equal;
xlim([-2.4 1.2]);
ylim([-2.9 0.6]);
legend('perturbed','target','controlled','FontSize',14,'FontName','Helvetica','Location','NorthWest');
xlabel('x','FontSize',16,'FontName','Helvetica');
ylabel('y','FontSize',16,'FontName','Helvetica');
set(gca,'FontSize',16,'FontName','Helvetica');
set(gca,'XTick',[-3 -2 -1 0 1]);
set(gca,'YTick',[-3 -2 -1 0]);
%set(gca,'XTickLabel',{'';'';''});
%set(gca,'YTickLabel',{'-\pi';'';'0';'';'\pi';'3\pi/2';'2\pi'});
%set(gca,'TickLength',[0.025 0.025]);